import os
import torch
import faiss
import numpy as np
from pathlib import Path
from collections import Counter
import json
import time

from app import (
    CNNEmbedder, encode_sequence, kmerize, jaccard, 
    KMER_K, JACCARD_THRESHOLD, TOP_K, CONF_THRESHOLD, Z_NOVELTY_THRESHOLD, DEVICE,
    CACHE_ROOT, MODEL_ROOT, INDEX_ROOT, open_fasta
)

class HierarchicalPipeline:
    def __init__(self, marker="16S"):
        self.marker = marker
        self.ranks = ["phylum", "class", "order", "family", "genus"]
        self.models = {}
        self.indexes = {}
        self.reference_kmers = []
        
        print("Loading pipeline assets...")
        self._load_deterministic()
        self._load_ml_cascade()
        print("Pipeline ready.")
        
    def _load_deterministic(self):
        fasta_path = CACHE_ROOT / self.marker / "family" / "train.fasta"
        if not fasta_path.exists():
            print(f"Warning: Reference fasta missing at {fasta_path}")
            return
            
        for rec in open_fasta(fasta_path):
            # Parse the taxonomy label if attached
            parts = rec.id.split("|")
            tax_label = parts[-1] if len(parts) > 1 else rec.id
            self.reference_kmers.append((tax_label, kmerize(str(rec.seq), KMER_K)))
            
    def _load_ml_cascade(self):
        for rank in self.ranks:
            # Load model
            model_path = MODEL_ROOT / self.marker / rank / "embedding.pt"
            if not model_path.exists():
                print(f"Warning: Missing model for {rank} at {model_path}")
                continue
                
            model = CNNEmbedder().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            self.models[rank] = model
            
            # Load index
            idx_dir = INDEX_ROOT / self.marker / rank
            if not (idx_dir / "index.faiss").exists():
                print(f"Warning: Missing FAISS index for {rank} at {idx_dir}")
                continue
                
            idx = faiss.read_index(str(idx_dir / "index.faiss"))
            ids = np.load(idx_dir / "ids.npy", allow_pickle=True)
            
            stats = {"mean": 0.9, "std": 0.05}
            meta_path = idx_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    stats.update(json.load(f))
                    
            self.indexes[rank] = (idx, ids, stats)
            
    def predict(self, seq: str):
        # 1. Deterministic Stage
        t0 = time.time()
        query_k = kmerize(seq, KMER_K)
        for rid, rk in self.reference_kmers:
            if jaccard(query_k, rk) >= JACCARD_THRESHOLD:
                return {
                    "prediction": "KNOWN_EXACT",
                    "level": "species",
                    "matched_label": rid,
                    "path": "deterministic",
                    "time": time.time() - t0
                }
                
        # 2. Hierarchical ML Cascade
        X = encode_sequence(seq)
        
        for rank in self.ranks:
            if rank not in self.models or rank not in self.indexes:
                continue
                
            model = self.models[rank]
            idx, ids, stats = self.indexes[rank]
            
            with torch.no_grad():
                emb = model(X).cpu().numpy()
            faiss.normalize_L2(emb)
            
            D, I = idx.search(emb, TOP_K)
            hits = ids[I[0]].tolist()
            dists = D[0].tolist()
            
            label, freq = Counter(hits).most_common(1)[0]
            conf = freq / TOP_K
            mean_dist = float(np.mean(dists))
            z = (mean_dist - stats["mean"]) / stats["std"]
            
            # Decision boundary (tuned to 1.5 for stricter novelty gating)
            if conf < CONF_THRESHOLD or z >= 1.5:
                return {
                    "prediction": f"NOVEL_{rank.upper()}",
                    "level": rank,
                    "confidence": conf,
                    "z_score": z,
                    "path": "ml_cascade",
                    "time": time.time() - t0
                }
                
        # If it passes all 5 ranks without failing a novelty check, it's known at the genus level!
        return {
            "prediction": "KNOWN_GENUS",
            "level": "genus",
            "matched_label": label,
            "confidence": conf,
            "path": "ml_cascade",
            "time": time.time() - t0
        }

if __name__ == "__main__":
    pipeline = HierarchicalPipeline()
    # Dummy test
    res = pipeline.predict("ACGTACGTACGTACGTACGTACGT")
    print(res)
