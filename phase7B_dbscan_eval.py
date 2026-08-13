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
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

from app import encode_sequence, CNNEmbedder, DEVICE

base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

def calc_purity(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def calc_intersection_stability(X, labels, eps, min_samples, metric, n_iters=100):
    n_samples = X.shape[0]
    stabilities = []
    
    for _ in range(n_iters):
        sub_idx = np.random.choice(n_samples, int(0.8 * n_samples), replace=False)
        X_sub = X[sub_idx]
        db_sub = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        sub_labels = db_sub.fit_predict(X_sub)
        
        orig_labels_sub = labels[sub_idx]
        
        ari = adjusted_rand_score(orig_labels_sub, sub_labels)
        stabilities.append(ari)
        
    return np.mean(stabilities), np.percentile(stabilities, [2.5, 97.5])

def run_null_permutation(y_true, y_pred, n_iters=1000):
    aris = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    for _ in range(n_iters):
        # Permute true labels
        shuffled_true = np.random.permutation(y_true_arr)
        aris.append(adjusted_rand_score(shuffled_true, y_pred_arr))
        
    aris = np.array(aris)
    mean_null = np.mean(aris)
    sd_null = np.std(aris)
    ci_null = np.percentile(aris, [2.5, 97.5])
    
    return mean_null, sd_null, ci_null, aris

def process_test_config(args):
    rank, seed = args
    config_key = f"{rank}_{seed}"
    print(f"[{config_key}] Running Phase 7B Evaluation...")
    
    with open(f"{base_dir}/dbscan_hyperparameters.json", "r") as f:
        hyperparams = json.load(f)
        
    if config_key not in hyperparams:
        print(f"[{config_key}] No parameters found!")
        return None
        
    params = hyperparams[config_key]
    eps = params['eps']
    min_samples = params['min_samples']
    
    test_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/test.fasta"
    model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    
    # We evaluate ground truth clustering always against true FAMILY
    # (since the open-set holdout rank is primarily Family, or we just want taxonomy coherence)
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 5:
                tax_map[parts[0]] = parts[5] # Always evaluate against true Family label
                
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    
    index = faiss.read_index(f"{index_dir}/index.faiss")
    faiss_ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
    with open(f"{index_dir}/meta.json") as f:
        stats = json.load(f)
        
    test_records = list(SeqIO.parse(test_fa, "fasta"))
    total_test = len(test_records)
    
    cand_embs = []
    cand_true_families = []
    
    with torch.no_grad():
        for r in test_records:
            seq = str(r.seq).upper()
            seq_id = r.id.split("|")[0] # clean id if necessary
            
            X = encode_sequence(seq)
            emb = model(X).cpu().numpy()
            faiss.normalize_L2(emb)
            
            D, I = index.search(emb, 10)
            hits = faiss_ids[I[0]].tolist()
            
            # Predict NOVEL using standard Phase 5 frozen rule
            # Wait, the translation to taxonomy for inference needs the RANK specific labels
            # But the novelty rule depends on Conf (frequency of majority class in 10-NN)
            # We need the train_fit labels for this rank. 
            # In phase 6, `run_benchmark_phase6` mapped hits to the target rank.
            rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
            target_idx = rank_map.get(rank, 5)
            
            # Temporary mapping for this rank's predictions
            local_tax_map = {}
            with open(tax_file, "r", encoding="utf-8") as ft:
                for line in ft:
                    parts = line.strip().split("\t")
                    if len(parts) > target_idx:
                        local_tax_map[parts[0]] = parts[target_idx]
                        
            translated = [local_tax_map.get(h, h) for h in hits]
            freq = Counter(translated).most_common(1)[0][1]
            conf = freq / 10.0
            z = (float(D[0][0]) - stats["mean"]) / stats["std"]
            
            if z >= 1.0 or conf < 0.5:
                cand_embs.append(emb[0])
                cand_true_families.append(tax_map.get(seq_id, "UNKNOWN"))
                
    n_cands = len(cand_embs)
    
    if n_cands < 10:
        return None
        
    X_test = np.array(cand_embs)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(X_test)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_rate = n_noise / len(labels)
    
    cluster_sizes = [list(labels).count(c) for c in set(labels) if c != -1]
    largest_fraction = max(cluster_sizes) / len(labels) if cluster_sizes else 0
    median_size = np.median(cluster_sizes) if cluster_sizes else 0
    
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        sil = 0.0
        db_idx = 0.0
    else:
        sil = silhouette_score(X_test[mask], labels[mask], metric='cosine')
        db_idx = davies_bouldin_score(X_test[mask], labels[mask])
        
    # Taxonomic Alignment against TRUE FAMILY
    true_fams = np.array(cand_true_families)
    
    ari = adjusted_rand_score(true_fams, labels)
    nmi = normalized_mutual_info_score(true_fams, labels)
    purity = calc_purity(true_fams, labels)
    
    # Stability
    mean_stab, ci_stab = calc_intersection_stability(X_test, labels, eps, min_samples, 'cosine', n_iters=100)
    
    # Null Model
    mean_null, sd_null, ci_null, aris_null = run_null_permutation(true_fams, labels, n_iters=1000)
    pval = np.sum(aris_null >= ari) / 1000.0
    
    print(f"[{config_key}] Done. Cands: {n_cands}/{total_test}. N_clust: {n_clusters}. ARI: {ari:.3f} (p={pval:.3f})")
    
    return {
        'rank': rank,
        'seed': seed,
        'total_test_seqs': total_test,
        'novel_candidates': n_cands,
        'candidate_fraction': n_cands / total_test,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_fraction': noise_rate,
        'largest_cluster_fraction': largest_fraction,
        'median_cluster_size': median_size,
        'silhouette': sil,
        'davies_bouldin': db_idx,
        'ari': ari,
        'nmi': nmi,
        'purity': purity,
        'stability_mean': mean_stab,
        'stability_2.5': ci_stab[0],
        'stability_97.5': ci_stab[1],
        'null_ari_mean': mean_null,
        'null_ari_sd': sd_null,
        'null_ari_2.5': ci_null[0],
        'null_ari_97.5': ci_null[1],
        'permutation_p_value': pval
    }

def main():
    print("Phase 7B: Locked Final Evaluation")
    
    tasks = [(r, s) for r in ranks for s in seeds]
    workers = max(1, os.cpu_count() - 2)
    
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(process_test_config, tasks):
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    df.to_csv(f"{base_dir}/phase7_final_metrics.csv", index=False)
    
    # Generate grouped summary
    summary = df.groupby('rank').mean().reset_index()
    summary.to_csv(f"{base_dir}/phase7_summary_by_rank.csv", index=False)
    
    print("Phase 7B Complete! Saved phase7_final_metrics.csv")

if __name__ == "__main__":
    main()
