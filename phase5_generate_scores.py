import os
import json
import hashlib
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import torch
import faiss
import time
import concurrent.futures

from app import encode_sequence, CNNEmbedder, DEVICE
from src.baseline_corrected import ClassicalBaselineCorrected

base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"

def get_hash(file_path):
    if not os.path.exists(file_path): return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_config_hash(rank, seed):
    return hashlib.md5(f"{rank}_{seed}".encode()).hexdigest()

def load_taxonomy_map(target_rank: str, source_rank="family"):
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(target_rank.lower(), 5)
    source_idx = rank_map.get(source_rank.lower(), 5)
    
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
                tax_map[parts[source_idx]] = parts[target_idx]
    return tax_map

def score_supcon(seq, model, index, ids, stats, tax_map):
    X = encode_sequence(seq)
    with torch.no_grad():
        emb = model(X).cpu().numpy()
        
    faiss.normalize_L2(emb)
    D, I = index.search(emb, 10)
    
    hits = ids[I[0]].tolist()
    dists = D[0].tolist()
    
    translated_hits = [tax_map.get(hit, hit) for hit in hits]
    label, freq = Counter(translated_hits).most_common(1)[0]
    
    conf = freq / 10.0
    mean_dist = float(dists[0])
    z = (mean_dist - stats["mean"]) / stats["std"]
    
    return label, mean_dist, z, conf

def generate_scores_for_config(args):
    rank, seed = args
    print(f"[{rank} | seed {seed}] Starting Score Generation...")
    
    fit_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
    eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
    val_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/val.fasta"
    
    model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    
    fit_hash = get_hash(fit_fa)
    eval_hash = get_hash(eval_fa)
    val_hash = get_hash(val_fa)
    
    tax_map = load_taxonomy_map(target_rank=rank)
    
    # Load sequences
    fit_records = list(SeqIO.parse(fit_fa, "fasta"))
    eval_records = list(SeqIO.parse(eval_fa, "fasta"))
    val_records = list(SeqIO.parse(val_fa, "fasta"))
    
    fit_taxa = set(tax_map.get(r.id.split('|')[0], "Unknown") for r in fit_records)
    
    eval_records = [r for r in eval_records if tax_map.get(r.id.split('|')[0], "Unknown") in fit_taxa]
    eval_taxa = set(tax_map.get(r.id.split('|')[0], "Unknown") for r in eval_records)
    val_records = [r for r in val_records if tax_map.get(r.id.split('|')[0], "Unknown") not in fit_taxa]
    val_taxa = set(tax_map.get(r.id.split('|')[0], "Unknown") for r in val_records)
    
    # 1. RIGOROUS DATASET ASSERTIONS
    assert eval_taxa.issubset(fit_taxa), f"Taxa Leak: {rank}_{seed} Eval not subset of Fit!"
    assert len(val_taxa.intersection(fit_taxa)) == 0, f"Taxa Leak: {rank}_{seed} Val overlaps with Fit!"
    
    # 2. Train Corrected Baseline (Strictly on train_fit)
    baseline = ClassicalBaselineCorrected(k=4)
    baseline.train(fit_fa, tax_file, rank)
    
    # 3. Load SupCon & FAISS
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    
    index = faiss.read_index(f"{index_dir}/index.faiss")
    faiss_ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
    with open(f"{index_dir}/meta.json") as f:
        stats = json.load(f)
        
    provenance = {
        "experiment_id": f"{rank}_{seed}",
        "rank": rank,
        "seed": seed,
        "distance_reference_n": stats.get("num_vectors", len(fit_records)),
        "mu": stats["mean"],
        "sigma": stats["std"],
        "reference_dataset_hash": fit_hash,
        "baseline_probability_source": "MultinomialNB_max_P(y|x)",
        "train_fit_hash": fit_hash,
        "train_eval_hash": eval_hash,
        "val_hash": val_hash
    }
    
    raw_scores = []
    
    # Helper to process datasets
    def process_dataset(records, is_novel):
        seq_ids = [r.id.split('|')[0] for r in records]
        seqs = [str(r.seq).upper() for r in records]
        y_true = ["NOVEL" if is_novel else tax_map.get(sid, "Unknown") for sid in seq_ids]
        
        # Baseline Inference
        X_base = baseline.vectorizer.transform(seqs)
        probs = baseline.model.predict_proba(X_base)
        max_probs = np.max(probs, axis=1)
        best_class_idx = np.argmax(probs, axis=1)
        
        for i, (sid, seq, yt, max_p, b_idx) in enumerate(zip(seq_ids, seqs, y_true, max_probs, best_class_idx)):
            # SupCon Inference
            s_label, s_dist, s_z, s_conf = score_supcon(seq, model, index, faiss_ids, stats, tax_map)
            
            raw_scores.append({
                "sequence_id": sid,
                "rank": rank,
                "seed": seed,
                "true_status": "NOVEL" if is_novel else "KNOWN",
                "true_taxon": yt,
                "model": "SupCon",
                "distance": s_dist,
                "z_score": s_z,
                "confidence": s_conf,
                "entropy": None,
                "predicted_taxon": s_label
            })
            
            raw_scores.append({
                "sequence_id": sid,
                "rank": rank,
                "seed": seed,
                "true_status": "NOVEL" if is_novel else "KNOWN",
                "true_taxon": yt,
                "model": "ClassicalBaseline",
                "distance": None,
                "z_score": None,
                "confidence": float(max_p),
                "entropy": None,
                "predicted_taxon": baseline.model.classes_[b_idx]
            })
            
    # Process KNOWN and NOVEL sets
    process_dataset(eval_records, is_novel=False)
    process_dataset(val_records, is_novel=True)
    
    return raw_scores, provenance

def main():
    print("PHASE 5: Generating Raw Novelty Scores")
    
    tasks = [(r, s) for r in ranks for s in seeds]
    workers = max(1, os.cpu_count() - 1)
    
    all_scores = []
    all_provenance = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for scores, prov in executor.map(generate_scores_for_config, tasks):
            all_scores.extend(scores)
            all_provenance.append(prov)
            
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(f"{base_dir}/phase5_raw_scores.csv", index=False)
    
    with open(f"{base_dir}/phase5_score_provenance.json", "w") as f:
        json.dump(all_provenance, f, indent=4)
        
    print(f"Generated scores for {len(all_scores)} samples. Saved to phase5_raw_scores.csv.")

if __name__ == "__main__":
    main()
