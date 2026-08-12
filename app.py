import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from huggingface_hub import snapshot_download
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from collections import Counter
from datetime import datetime
import gzip
import json
import io
import uuid
import time
import logging

import numpy as np
import torch
import faiss
from Bio import SeqIO

from src.routing import MarkerDetector
from src.qc import sequence_qc


# ========== CONFIG ==========
KMER_K = 21
JACCARD_THRESHOLD = 0.95
TOP_K = 10
CONF_THRESHOLD = 0.6
Z_NOVELTY_THRESHOLD = 2.0
MAX_SEQ_LEN = 500
VOCAB_SIZE = 4096  # 4^6 for k=6

MODEL_ROOT = Path("./assets/HTRP_models/embedding")
INDEX_ROOT = Path("./assets/HTRP_indexes")
CACHE_ROOT = Path("./assets/HTRP_cache")
HISTORY_FILE = Path("analysis_history.jsonl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANKS = ["family", "genus"]

VALID_MARKERS = {"16S", "18S", "CO1", "FITS", "PITS", "trnL", "Vert12S"}

app = FastAPI()


# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("HARBOUR")


# ========== HF ASSET BOOTSTRAP (robust) ==========
if not Path("./assets/HTRP_models").exists():
    logger.info("Downloading eDNA assets from Hugging Face...")
    snapshot_download(
        repo_id="chiraghariprasad/eDNA",
        repo_type="dataset",
        local_dir="./assets"
    )
    logger.info("Assets downloaded.")


# ========== GLOBAL CACHES (lazy loaded) ==========
MODELS = {}
INDEXES = {}
REFERENCE_KMERS = {}
MARKER_DETECTOR = None


# ========== MODEL DEFINITION ==========
class CNNEmbedder(torch.nn.Module):
    def __init__(self, embed_dim=128, conv_channels=128):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE + 1, 64, padding_idx=VOCAB_SIZE)
        self.conv1 = torch.nn.Conv1d(64, conv_channels, 5, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(conv_channels)
        self.conv2 = torch.nn.Conv1d(conv_channels, conv_channels, 5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(conv_channels)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc = torch.nn.Linear(conv_channels, embed_dim)

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return torch.nn.functional.normalize(self.fc(x), dim=1)


# ========== UTILITY FUNCTIONS ==========
def log_history(entry: dict):
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def open_fasta(path: Path):
    if path.suffix == ".gz":
        return SeqIO.parse(gzip.open(path, "rt", encoding="utf-8", errors="ignore"), "fasta")
    return SeqIO.parse(path, "fasta")

def kmerize(seq: str, k: int):
    seq = seq.upper()
    return {
        seq[i:i+k]
        for i in range(len(seq) - k + 1)
        if "N" not in seq[i:i+k]
    }

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)

def encode_kmer(kmer: str) -> int:
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = 0
    for char in kmer:
        if char not in mapping:
            return VOCAB_SIZE  # Map Ns or invalid chars to padding
        idx = idx * 4 + mapping[char]
    return idx

def encode_sequence(seq: str, k=6):
    seq = seq.upper()
    tokens = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if "N" not in kmer:
            tokens.append(encode_kmer(kmer))
        if len(tokens) >= MAX_SEQ_LEN:
            break
            
    # Pad to MAX_SEQ_LEN
    while len(tokens) < MAX_SEQ_LEN:
        tokens.append(VOCAB_SIZE)
            
    X = torch.tensor([tokens], dtype=torch.long)
    return X.to(DEVICE)

def build_reference_kmers(marker: str):
    fasta = CACHE_ROOT / marker / "family" / "subsampled.fasta"
    ref = []
    for rec in open_fasta(fasta):
        ref.append((rec.id, kmerize(str(rec.seq), KMER_K)))
    return ref

def load_model(marker: str):
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_ROOT / marker / "embedding.pt", map_location=DEVICE)
    )
    model.eval()
    return model

def load_index(marker: str, rank: str):
    idx = faiss.read_index(str(INDEX_ROOT / marker / rank / "index.faiss"))
    ids = np.load(INDEX_ROOT / marker / rank / "ids.npy", allow_pickle=True)

    meta = INDEX_ROOT / marker / rank / "meta.json"
    stats = {"mean": 0.9, "std": 0.05}
    if meta.exists():
        stats.update(json.loads(meta.read_text()))

    return idx, ids, stats

# ----- Lazy loading wrappers -----
def get_model(marker: str):
    if marker not in MODELS:
        MODELS[marker] = load_model(marker)
    return MODELS[marker]

def get_reference_kmers(marker: str):
    if marker not in REFERENCE_KMERS:
        REFERENCE_KMERS[marker] = build_reference_kmers(marker)
    return REFERENCE_KMERS[marker]

def get_index(marker: str, rank: str):
    key = (marker, rank)
    if key not in INDEXES:
        INDEXES[key] = load_index(marker, rank)
    return INDEXES[key]


# ========== DETERMINISTIC MATCHER ==========
def deterministic_match(seq: str, marker: str):
    query_k = kmerize(seq, KMER_K)
    for rid, rk in get_reference_kmers(marker):
        score = jaccard(query_k, rk)
        if score >= JACCARD_THRESHOLD:
            return {
                "reference_id": rid,
                "jaccard": round(score, 4),
                "query_kmers": len(query_k),
                "ref_kmers": len(rk),
                "matched_kmers": len(query_k & rk)
            }
    return None


# ========== ML + FAISS FALLBACK ==========
def infer_ml(seq: str, marker: str):
    model = get_model(marker)
    X = encode_sequence(seq)

    with torch.no_grad():
        emb = model(X).cpu().numpy()
    faiss.normalize_L2(emb)

    results = {}
    for rank in RANKS:
        index, ids, stats = get_index(marker, rank)
        D, I = index.search(emb, TOP_K)

        hits = ids[I[0]].tolist()
        dists = D[0].tolist()

        neighbors = [
            {"id": hits[i], "distance": round(dists[i], 4)}
            for i in range(len(hits))
        ]

        label, freq = Counter(hits).most_common(1)[0]
        conf = freq / TOP_K
        mean_dist = float(np.mean(dists))
        z = (mean_dist - stats["mean"]) / stats["std"]

        votes = [
            {
                "label": lbl,
                "count": cnt,
                "percentage": round(cnt / TOP_K * 100, 1)
            }
            for lbl, cnt in Counter(hits).most_common()
        ]

        prediction = (
            label if conf >= CONF_THRESHOLD and z < Z_NOVELTY_THRESHOLD
            else "NOVEL"
        )

        results[rank] = {
            "prediction": prediction,
            "confidence": round(conf, 4),
            "mean_distance": round(mean_dist, 4),
            "z_score": round(z, 3),
            "neighbors": neighbors,
            "votes": votes,
            "stats": stats,
            "thresholds": {
                "confidence": CONF_THRESHOLD,
                "z_score": Z_NOVELTY_THRESHOLD
            }
        }

        if prediction == "NOVEL":
            break

    return results


# ========== PIPELINE ENTRY ==========
def run_pipeline(seq_id: str, seq: str, marker: str):
    global MARKER_DETECTOR
    run_id = str(uuid.uuid4())
    t0 = time.time()

    logger.info(f"[{run_id}] START → {seq_id}")
    
    # 1. Quality Control Gate
    is_corrupted, qc_details = sequence_qc(seq)
    if is_corrupted:
        return {
            "run_id": run_id,
            "query": seq_id,
            "path": "qc",
            "prediction": "CORRUPTED",
            "details": qc_details,
            "timing": {"total": round(time.time() - t0, 3)}
        }
        
    # 2. Marker Routing / Verification
    if MARKER_DETECTOR is None:
        MARKER_DETECTOR = MarkerDetector(Path("./assets/HTRP_models/routing/marker_model.pkl"))
        
    detected_marker = MARKER_DETECTOR.detect(seq)
    # Use detected marker if the user didn't specify one or specified 'auto', 
    # but for now we'll just log it or override if it's confident
    if marker == "auto" or marker not in VALID_MARKERS:
        marker = detected_marker["label"]

    t_det = time.time()
    hit = deterministic_match(seq, marker)
    det_time = round(time.time() - t_det, 3)

    if hit:
        result = {
            "run_id": run_id,
            "query": seq_id,
            "path": "deterministic",
            "prediction": "KNOWN",
            "confidence": 1.0,
            "details": hit,
            "timing": {"deterministic": det_time, "ml": 0.0}
        }
    else:
        t_ml = time.time()
        ml_results = infer_ml(seq, marker)
        ml_time = round(time.time() - t_ml, 3)

        final_prediction = "KNOWN"
        for r in ml_results.values():
            if r["prediction"] == "NOVEL":
                final_prediction = "NOVEL"
                break

        result = {
            "run_id": run_id,
            "query": seq_id,
            "path": "ml",
            "prediction": final_prediction,
            "results": ml_results,
            "timing": {"deterministic": det_time, "ml": ml_time}
        }

    total_time = round(time.time() - t0, 3)
    result["timing"]["total"] = total_time

    logger.info(f"[{run_id}] DONE → {result['prediction']} ({total_time}s)")

    log_history({
        "timestamp": datetime.utcnow().isoformat(),
        "marker": marker,
        "result": result
    })

    return result


# ========== FASTAPI ROUTES ==========
@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve a simple HTML form if index.html is missing
    index_path = Path("index.html")
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    # Fallback minimal UI
    return """
    <html>
        <body>
            <h2>HARBOUR eDNA Classifier</h2>
            <form action="/infer" method="post">
                Marker: <select name="marker">
                    <option>16S</option><option>18S</option><option>CO1</option>
                    <option>FITS</option><option>PITS</option><option>trnL</option>
                    <option>Vert12S</option>
                </select><br>
                Sequence: <textarea name="sequence" rows="4" cols="50"></textarea><br>
                <input type="submit" value="Classify">
            </form>
            <hr>
            <form action="/infer_batch" method="post" enctype="multipart/form-data">
                Marker: <select name="marker">...</select><br>
                FASTA file: <input type="file" name="fasta"><br>
                <input type="submit" value="Batch classify">
            </form>
        </body>
    </html>
    """

@app.post("/infer")
async def infer_single(sequence: str = Form(...), marker: str = Form(...)):
    if marker != "auto" and marker not in VALID_MARKERS:
        raise HTTPException(400, f"Invalid marker. Choose from {VALID_MARKERS} or 'auto'")
    return run_pipeline("query_sequence", sequence, marker)

@app.post("/infer_batch")
async def infer_batch(marker: str = Form(...), fasta: UploadFile = File(...)):
    if marker != "auto" and marker not in VALID_MARKERS:
        raise HTTPException(400, f"Invalid marker. Choose from {VALID_MARKERS} or 'auto'")
    content = await fasta.read()
    records = list(SeqIO.parse(io.StringIO(content.decode()), "fasta"))

    total = len(records)
    logger.info(f"BATCH START → {total} sequences")

    results = []
    for i, rec in enumerate(records, 1):
        logger.info(f"BATCH PROGRESS → {i}/{total}")
        results.append(run_pipeline(rec.id, str(rec.seq), marker))

    return {
        "summary": {
            "total": total,
            "known": sum(r["prediction"] == "KNOWN" for r in results),
            "novel": sum(r["prediction"] == "NOVEL" for r in results),
            "corrupted": sum(r["prediction"] == "CORRUPTED" for r in results),
            "deterministic": sum(r["path"] == "deterministic" for r in results),
            "ml_based": sum(r["path"] == "ml" for r in results)
        },
        "results": results
    }

@app.get("/history")
async def history(limit: int = 50):
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open() as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in reversed(lines)]


# ========== MAIN ==========
if __name__ == "__main__":
    import uvicorn
    # For HF Spaces, the Dockerfile will override port to 7860 if needed
    uvicorn.run(app, host="0.0.0.0", port=8000)