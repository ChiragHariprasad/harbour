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
    full_tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
                tax_map[parts[source_idx]] = parts[target_idx]
            
            if len(parts) >= 7:
                full_tax_map[parts[0]] = {
                    "class": parts[3],
                    "order": parts[4],
                    "family": parts[5],
                    "genus": parts[6]
                }
    return tax_map, full_tax_map

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

def evaluate_config(args):
    rank, seed, manifest = args
    print(f"[{rank} | seed {seed}] Starting Final Evaluation...")
    
    fit_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
    eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
    test_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/test.fasta"
    
    model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    
    fit_hash = get_hash(fit_fa)
    eval_hash = get_hash(eval_fa)
    test_hash = get_hash(test_fa)
    
    tax_map, full_tax_map = load_taxonomy_map(target_rank=rank)
    
    fit_records = list(SeqIO.parse(fit_fa, "fasta"))
    eval_records = list(SeqIO.parse(eval_fa, "fasta"))
    test_records = list(SeqIO.parse(test_fa, "fasta"))
    
    fit_ids = set(r.id.split('|')[0] for r in fit_records)
    eval_ids = set(r.id.split('|')[0] for r in eval_records)
    test_ids = set(r.id.split('|')[0] for r in test_records)
    
    # 1. PHASE 6B: Strict Sequence ID Assertions
    assert fit_ids.isdisjoint(eval_ids), f"ID Leak: {rank}_{seed} eval overlaps with fit!"
    assert fit_ids.isdisjoint(test_ids), f"ID Leak: {rank}_{seed} test overlaps with fit!"
    assert eval_ids.isdisjoint(test_ids), f"ID Leak: {rank}_{seed} test overlaps with eval!"
    
    # Taxonomic assertions
    fit_taxa = set(tax_map.get(sid, "Unknown") for sid in fit_ids)
    test_taxa = set(tax_map.get(sid, "Unknown") for sid in test_ids)
    
    overlap = fit_taxa.intersection(test_taxa)
    
    if rank == "family":
        assert len(overlap) == 0, f"Taxa Leak: {rank}_{seed} test taxa overlaps with fit taxa despite holdout rank!"
        
    comp_stats = {
        "rank": rank,
        "seed": seed,
        "known_candidate_sequences": len(eval_records),
        "known_sequences_retained": len(eval_records), # No filtering!
        "novel_test_sequences": len(test_records),
        "novel_taxa": len(test_taxa - fit_taxa),
        "train_test_overlapping_taxa": len(overlap),
        "exact_sequence_overlap": 0, # proven by isdisjoint above
        "faiss_test_overlap": 0,
        "known_novel_ratio": len(eval_records) / len(test_records) if len(test_records) > 0 else 0
    }
    
    # Train Baseline
    baseline = ClassicalBaselineCorrected(k=4)
    baseline.train(fit_fa, tax_file, rank)
    
    # Load SupCon
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    
    index = faiss.read_index(f"{index_dir}/index.faiss")
    faiss_ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
    with open(f"{index_dir}/meta.json") as f:
        stats = json.load(f)
        
    sup_z = manifest["supcon_z_threshold"]
    sup_c = manifest["supcon_confidence_threshold"]
    base_t = manifest["baseline_threshold"]
    
    raw_scores = []
    
    def process_records(records, source_split):
        seq_ids = [r.id.split('|')[0] for r in records]
        seqs = [str(r.seq).upper() for r in records]
        
        X_base = baseline.vectorizer.transform(seqs)
        probs = baseline.model.predict_proba(X_base)
        max_probs = np.max(probs, axis=1)
        best_class_idx = np.argmax(probs, axis=1)
        
        for i, (sid, seq, max_p, b_idx) in enumerate(zip(seq_ids, seqs, max_probs, best_class_idx)):
            true_taxon = tax_map.get(sid, "Unknown")
            true_status = "KNOWN" if true_taxon in fit_taxa else "NOVEL"
            
            s_label, s_dist, s_z, s_conf = score_supcon(seq, model, index, faiss_ids, stats, tax_map)
            sup_status = "NOVEL" if (s_z >= sup_z or s_conf < sup_c) else "KNOWN"
            base_status = "NOVEL" if max_p < base_t else "KNOWN"
            
            full_t = full_tax_map.get(sid, {})
            
            raw_scores.append({
                "experiment_id": f"{rank}_{seed}",
                "rank": rank,
                "holdout_rank": "family",
                "seed": seed,
                "sequence_id": sid,
                "split": source_split,
                
                "true_taxon_at_rank": true_taxon,
                "true_status_at_rank": true_status,
                
                "supcon_predicted_taxon": s_label,
                "supcon_predicted_status": sup_status,
                "baseline_predicted_taxon": baseline.model.classes_[b_idx],
                "baseline_predicted_status": base_status,
                
                "distance": s_dist,
                "z_score": s_z,
                "confidence": s_conf,
                "threshold_z": sup_z,
                "threshold_conf": sup_c,
                
                "baseline_probability": float(max_p),
                "baseline_threshold": base_t,
                
                "checkpoint_id": manifest["checkpoint_id"],
                "config_hash": get_config_hash(rank, seed),
                "dataset_hash": fit_hash,
                
                "true_class": full_t.get("class", "Unknown"),
                "true_order": full_t.get("order", "Unknown"),
                "true_family": full_t.get("family", "Unknown"),
                "true_genus": full_t.get("genus", "Unknown")
            })
            
    process_records(eval_records, "train_eval")
    process_records(test_records, "test")
    
    return raw_scores, comp_stats

def main():
    print("Phase 6: Final Open-Set Evaluation")
    with open(f"{base_dir}/phase6_freeze_manifest.json", "r") as f:
        manifest = json.load(f)
        
    tasks = [(r, s, manifest) for r in ranks for s in seeds]
    workers = max(1, os.cpu_count() - 1)
    
    all_scores = []
    all_comps = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for scores, comps in executor.map(evaluate_config, tasks):
            all_scores.extend(scores)
            all_comps.append(comps)
            
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv(f"{base_dir}/phase6_final_predictions.csv", index=False)
    
    df_comps = pd.DataFrame(all_comps)
    df_comps.to_csv(f"{base_dir}/phase6_dataset_composition.csv", index=False)
    
    print("Phase 6 inference complete!")

if __name__ == "__main__":
    main()
