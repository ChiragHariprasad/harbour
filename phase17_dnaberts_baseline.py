import os
import json
import time
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import faiss
import concurrent.futures

BASE_DIR = "E:/HTRP"
OUTPUT_BASE = "E:/paper_revamp/phase17_dnaberts_baseline"

# 1. Load the unified corpus embeddings
def load_corpus(output_base):
    print("Loading unified DNABERT-S corpus embeddings...")
    emb_path = f"{output_base}/corpus_embeddings.npy"
    ids_path = f"{output_base}/corpus_ids.npy"
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found at {emb_path}. Run phase17_dnaberts_embed.py first.")
        
    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    
    faiss.normalize_L2(embeddings)
    
    print(f"Loaded {len(ids)} sequences of dimension {embeddings.shape[1]}")
    # Create lookup map for fast retrieval
    lookup = {str(seq_id).split('|')[0]: idx for idx, seq_id in enumerate(ids)}
    return embeddings, ids, lookup

def load_taxonomy_map(target_rank: str, source_rank="family"):
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(target_rank.lower(), 5)
    source_idx = rank_map.get(source_rank.lower(), 5)
    
    tax_map = {}
    tax_file = f"{BASE_DIR}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
    return tax_map

def compute_metrics(y_true, predictions, tau, closed_set_labels):
    TP = 0; FP = 0; TN = 0; FN = 0; correct_known = 0; total_known = 0
    for true_lbl, (pred_lbl, dist, z, conf) in zip(y_true, predictions):
        novel_pred = conf < tau
        is_novel = true_lbl not in closed_set_labels
        
        if is_novel and novel_pred: TN += 1
        elif is_novel and not novel_pred: FP += 1
        elif not is_novel and novel_pred: FN += 1
        else: TP += 1
        
        if not is_novel:
            total_known += 1
            if not novel_pred and pred_lbl == true_lbl:
                correct_known += 1
                
    A_known = correct_known / total_known if total_known > 0 else 0
    R_novel = TN / (TN + FP) if (TN + FP) > 0 else 0
    H = (2 * A_known * R_novel) / (A_known + R_novel) if (A_known + R_novel) > 0 else 0
    return H, A_known, R_novel

def evaluate_config(rank, seed, corpus_emb, corpus_lookup):
    t0 = time.time()
    
    tax_map = load_taxonomy_map(target_rank=rank)
    cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    
    fit_records = list(SeqIO.parse(f"{cache_dir}/train_fit.fasta", "fasta"))
    eval_records = list(SeqIO.parse(f"{cache_dir}/train_eval.fasta", "fasta"))
    val_records = list(SeqIO.parse(f"{cache_dir}/val.fasta", "fasta"))
    test_records = list(SeqIO.parse(f"{cache_dir}/test.fasta", "fasta"))
    
    # Identify training subset
    fit_accs = [r.id.split('|')[0] for r in fit_records]
    fit_indices = [corpus_lookup[a] for a in fit_accs if a in corpus_lookup]
    fit_emb = corpus_emb[fit_indices]
    fit_labels = [tax_map.get(a, "Unknown") for a in fit_accs if a in corpus_lookup]
    
    closed_set_labels = set(fit_labels)
    
    dim = fit_emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(fit_emb)
    
    # Calculate Novelty Stats (Mean/Std of top-1 distance on training)
    index.hnsw.efSearch = 64
    D, I = index.search(fit_emb, 2)
    nearest_dists = D[:, 1]
    stats_mean = float(np.mean(nearest_dists))
    stats_std = float(np.std(nearest_dists))
    
    def process_split(records):
        accs = [r.id.split('|')[0] for r in records]
        indices = [corpus_lookup[a] for a in accs if a in corpus_lookup]
        emb = corpus_emb[indices]
        true_labels = [tax_map.get(a, "Unknown") for a in accs if a in corpus_lookup]
        
        D_q, I_q = index.search(emb, 10)
        predictions = []
        for i in range(len(emb)):
            hits = [fit_labels[idx] for idx in I_q[i]]
            dists = D_q[i]
            
            label, freq = Counter(hits).most_common(1)[0]
            conf = freq / 10.0
            mean_dist = float(dists[0])
            z = (mean_dist - stats_mean) / stats_std
            
            predictions.append((label, mean_dist, z, conf))
        return accs, true_labels, predictions
        
    val_q_accs, val_true, val_preds = process_split(eval_records + val_records)
    
    # 1. Base Accuracy on Eval (Knowns only)
    eval_accs, eval_true, eval_preds = process_split(eval_records)
    correct_known_base = sum(1 for t, p in zip(eval_true, eval_preds) if t == p[0])
    A_cs = correct_known_base / len(eval_true) if len(eval_true) > 0 else 0
    
    # Threshold sweep on Validation Data
    best_H = -1; best_tau = 0
    for tau in np.arange(0.0, 1.05, 0.05):
        H, A_k, R_n = compute_metrics(val_true, val_preds, tau, closed_set_labels)
        if A_k >= 0.8 * A_cs and H > best_H:
            best_H = H
            best_tau = tau
            
    # Final Test Evaluation
    test_q_accs, test_true, test_preds = process_split(eval_records + test_records)
    H, A_k, R_n = compute_metrics(test_true, test_preds, best_tau, closed_set_labels)
    
    # Save predictions
    out_dir = f"{OUTPUT_BASE}/{rank}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({
        'accession': test_q_accs,
        'true_label': test_true,
        'pred_label': [p[0] for p in test_preds],
        'distance': [p[1] for p in test_preds],
        'z_score': [p[2] for p in test_preds],
        'confidence': [p[3] for p in test_preds],
        'is_novel': [p[3] < best_tau for p in test_preds]
    })
    df.to_csv(f"{out_dir}/predictions.csv", index=False)
    
    wall_time = time.time() - t0
    print(f"[{rank}/{seed}] Completed. H={H:.4f} A_known={A_k:.4f} R_novel={R_n:.4f} tau={best_tau*100:.2f}% Time={wall_time:.1f}s")
    
    return {
        'rank': rank, 'seed': seed, 
        'H_score': H, 'A_known': A_k, 'R_novel': R_n, 
        'tau': best_tau, 'wall_time': wall_time
    }

def main():
    corpus_emb, corpus_ids, corpus_lookup = load_corpus(OUTPUT_BASE)
    
    ranks = ['class', 'order', 'family', 'genus']
    seeds = [42, 43, 44, 45, 46]
    
    tasks = [(r, s) for r in ranks for s in seeds]
    results = []
    
    print("\nStarting Phase 17 DNABERT-S Baseline Evaluation")
    t_start = time.time()
    
    # Since FAISS is thread-safe and this is CPU bound but index search is C++,
    # we can use ThreadPoolExecutor to parallelize configurations
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_config, r, s, corpus_emb, corpus_lookup): (r, s) for r, s in tasks}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    print(f"\nFull evaluation completed in {time.time() - t_start:.2f}s")
    
    df = pd.DataFrame(results)
    
    print("\nAggregated Results:")
    for rank in ranks:
        sub = df[df['rank'] == rank]
        h_mean = sub['H_score'].mean()
        h_std = sub['H_score'].std()
        ak_mean = sub['A_known'].mean()
        rn_mean = sub['R_novel'].mean()
        print(f"{rank:10s} H={h_mean:.4f}+-{h_std:.4f} A_known={ak_mean:.4f} R_novel={rn_mean:.4f}")
        
    df.to_csv(f"{OUTPUT_BASE}/dnaberts_aggregate_metrics.csv", index=False)
    
if __name__ == "__main__":
    main()
