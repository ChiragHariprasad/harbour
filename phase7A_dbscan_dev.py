import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import concurrent.futures
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from app import encode_sequence, CNNEmbedder, DEVICE

base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

def calc_bootstrap_stability(X, labels, eps, min_samples, metric, n_iters=100):
    n_samples = X.shape[0]
    stabilities = []
    
    for _ in range(n_iters):
        # Sample with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)
        # Find unique observations in this bootstrap sample
        unique_idx = np.unique(idx)
        
        # We need to find observations that are in BOTH the original set and the bootstrap set.
        # Since unique_idx is a subset of the original indices, ALL unique_idx are common observations!
        X_boot = X[idx]
        
        # Run DBSCAN on bootstrap sample
        db_boot = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        boot_labels_full = db_boot.fit_predict(X_boot)
        
        # Map back to unique observations to compare with original labels
        # The first occurrence of each unique_idx in the idx array gives its cluster label
        _, first_occurence_indices = np.unique(idx, return_index=True)
        # However, np.unique sorts the unique values. So `unique_idx` corresponds to `idx[first_occurence_indices]`?
        # Actually, let's just do it simpler:
        # Instead of sampling with replacement for clustering, which creates duplicate identical rows that DBSCAN will just lump together...
        # Let's subsample WITHOUT replacement for stability, e.g., 80% of data.
        sub_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
        X_sub = X[sub_idx]
        db_sub = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        sub_labels = db_sub.fit_predict(X_sub)
        
        orig_labels_sub = labels[sub_idx]
        
        # Calculate ARI on the common observations
        ari = adjusted_rand_score(orig_labels_sub, sub_labels)
        stabilities.append(ari)
        
    return np.mean(stabilities)

def process_config(args):
    rank, seed = args
    print(f"[{rank}_{seed}] Starting DBSCAN Development Sweep...")
    
    val_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/val.fasta"
    model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    
    # Load tax map
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(rank, 5)
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > target_idx:
                tax_map[parts[0]] = parts[target_idx]
                
    # Load Model & FAISS
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    
    index = faiss.read_index(f"{index_dir}/index.faiss")
    faiss_ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
    with open(f"{index_dir}/meta.json") as f:
        stats = json.load(f)
        
    val_records = list(SeqIO.parse(val_fa, "fasta"))
    
    candidate_embeddings = []
    
    # 1. Identify NOVEL candidates using frozen Phase 5 rule
    with torch.no_grad():
        for r in val_records:
            seq = str(r.seq).upper()
            X = encode_sequence(seq)
            emb = model(X).cpu().numpy()
            faiss.normalize_L2(emb)
            
            D, I = index.search(emb, 10)
            hits = faiss_ids[I[0]].tolist()
            translated = [tax_map.get(h, h) for h in hits]
            
            freq = Counter(translated).most_common(1)[0][1]
            conf = freq / 10.0
            z = (float(D[0][0]) - stats["mean"]) / stats["std"]
            
            if z >= 1.0 or conf < 0.5:
                candidate_embeddings.append(emb[0])
                
    if len(candidate_embeddings) < 10:
        print(f"[{rank}_{seed}] Too few candidates ({len(candidate_embeddings)}).")
        return None
        
    X_cand = np.array(candidate_embeddings)
    
    # 2. Empirical k-NN Grid
    nn = NearestNeighbors(n_neighbors=10, metric='cosine')
    nn.fit(X_cand)
    distances, _ = nn.kneighbors(X_cand)
    
    # Extract distances to the 5th neighbor
    d5 = distances[:, 4]
    p10, p50, p90 = np.percentile(d5, [10, 50, 90])
    
    # Create eps grid around these percentiles
    eps_grid = np.linspace(max(0.01, p10 / 2), min(1.0, p90 * 2), num=15)
    min_samples_grid = [3, 5, 8, 12, 15]
    
    best_params = None
    best_stability = -1.0
    best_sil = -1.0
    best_db = float('inf')
    
    sweep_results = []
    
    # 3. Sweep
    for eps in eps_grid:
        for ms in min_samples_grid:
            db = DBSCAN(eps=eps, min_samples=ms, metric='cosine')
            labels = db.fit_predict(X_cand)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_rate = n_noise / len(labels)
            
            if n_clusters < 1:
                continue
                
            cluster_sizes = [list(labels).count(c) for c in set(labels) if c != -1]
            largest_fraction = max(cluster_sizes) / len(labels)
            median_size = np.median(cluster_sizes)
            
            # 4. Pathological Sanity Constraints
            if n_clusters < 2: continue
            if noise_rate <= 0 or noise_rate >= 0.30: continue
            if largest_fraction >= 0.95: continue
            if median_size < 3: continue
            
            # Calculate Silhouette excluding singletons/noise
            # Silhouette requires at least 2 clusters and >1 samples per cluster generally, but sklearn handles it if >1 labels.
            # We must filter out noise (-1) to evaluate the clustering quality of the clustered points!
            mask = labels != -1
            if len(set(labels[mask])) < 2: continue
            
            try:
                sil = silhouette_score(X_cand[mask], labels[mask], metric='cosine')
                db_idx = davies_bouldin_score(X_cand[mask], labels[mask])
            except:
                continue
                
            # Bootstrap stability
            stability = calc_bootstrap_stability(X_cand, labels, eps, ms, metric='cosine', n_iters=30)
            
            sweep_results.append({
                'eps': float(eps),
                'min_samples': int(ms),
                'n_clusters': n_clusters,
                'noise_rate': noise_rate,
                'largest_fraction': largest_fraction,
                'median_size': median_size,
                'stability': stability,
                'silhouette': sil,
                'davies_bouldin': db_idx
            })
            
            # 5. Lexicographic Selection: Stability -> Silhouette -> -DB
            # We use tolerances to avoid floating point strict inequality issues
            if stability > best_stability + 0.02:
                best_stability = stability
                best_sil = sil
                best_db = db_idx
                best_params = sweep_results[-1]
            elif abs(stability - best_stability) <= 0.02:
                if sil > best_sil + 0.02:
                    best_stability = stability
                    best_sil = sil
                    best_db = db_idx
                    best_params = sweep_results[-1]
                elif abs(sil - best_sil) <= 0.02:
                    if db_idx < best_db:
                        best_stability = stability
                        best_sil = sil
                        best_db = db_idx
                        best_params = sweep_results[-1]
                        
    if best_params is None:
        print(f"[{rank}_{seed}] WARNING: No parameters passed sanity constraints!")
        return None
        
    print(f"[{rank}_{seed}] Locked: eps={best_params['eps']:.4f}, min_samples={best_params['min_samples']}, N={best_params['n_clusters']}, Stability={best_params['stability']:.3f}, Noise={best_params['noise_rate']:.3f}")
    
    return {f"{rank}_{seed}": best_params}

def main():
    print("Phase 7A: DBSCAN Parameter Development")
    
    tasks = [(r, s) for r in ranks for s in seeds]
    workers = max(1, os.cpu_count() - 2)
    
    final_manifest = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(process_config, tasks):
            if res:
                final_manifest.update(res)
                
    with open(f"{base_dir}/dbscan_hyperparameters.json", "w") as f:
        json.dump(final_manifest, f, indent=4)
        
    print(f"Phase 7A Complete! Locked parameters for {len(final_manifest)} configs.")

if __name__ == "__main__":
    main()
