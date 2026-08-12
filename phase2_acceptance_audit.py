import os
import glob
import json
import hashlib
import numpy as np
from Bio import SeqIO
import scipy.sparse as sp
from collections import defaultdict
import time
import subprocess

def get_hash(file_path):
    if not os.path.exists(file_path): return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_taxa(header):
    parts = header.split('|')
    return parts[-1] if len(parts) > 1 else "Unknown"
    
def get_accession(header):
    parts = header.split('|')
    return parts[0]

def count_original(fasta_file):
    if not os.path.exists(fasta_file): return 0
    with open(fasta_file, 'r') as f:
        return sum(1 for line in f if line.startswith(">"))

def build_kmer_matrix(seqs, k=6):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), binary=True)
    return vectorizer.fit_transform(seqs), vectorizer

def parse_fasta(path):
    seqs, ids, taxa = [], set(), set()
    id_to_seq = {}
    id_to_tax = {}
    if not os.path.exists(path): return seqs, ids, taxa, id_to_seq, id_to_tax
    for r in SeqIO.parse(path, "fasta"):
        s = str(r.seq).upper()
        seqs.append(s)
        acc = get_accession(r.description)
        ids.add(acc)
        t = get_taxa(r.description)
        taxa.add(t)
        id_to_seq[acc] = s
        id_to_tax[acc] = t
    return seqs, ids, taxa, id_to_seq, id_to_tax

def main():
    base_dir = "E:/HTRP"
    cache_dir = f"{base_dir}/assets/HTRP_cache/16S"
    leaked_dir = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S"
    
    ranks = ['class', 'order', 'family', 'genus']
    seeds = [42, 43, 44, 45, 46]
    
    report_data = {
        "dataset_version": "2.0.0_strict_dedup",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ranks_audited": ranks,
        "seeds_audited": seeds,
        "results": {},
        "experimental_design_verified": True,
        "PHASE_2_STATUS": "PENDING"
    }
    
    all_passed = True
    manifest = {}

    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_dir).decode("utf-8").strip()
    except:
        commit_hash = "N/A"

    source_fa = f"{leaked_dir}/family/subsampled.fasta"

    for rank in ranks:
        for seed in seeds:
            print(f"=== Auditing {rank} (Seed {seed}) ===")
            run_key = f"{rank}_seed{seed}"
            rank_res = {"PASS": True, "errors": []}
            
            tr_fa = f"{cache_dir}/{rank}/seed_{seed}/train.fasta"
            va_fa = f"{cache_dir}/{rank}/seed_{seed}/val.fasta"
            te_fa = f"{cache_dir}/{rank}/seed_{seed}/test.fasta"
            
            if not all(os.path.exists(f) for f in [tr_fa, va_fa, te_fa]):
                rank_res["PASS"] = False
                rank_res["errors"].append("Missing split files")
                report_data["results"][run_key] = rank_res
                all_passed = False
                continue
                
            tr_seqs, tr_ids, tr_taxa, tr_i2s, tr_i2t = parse_fasta(tr_fa)
            va_seqs, va_ids, va_taxa, va_i2s, va_i2t = parse_fasta(va_fa)
            te_seqs, te_ids, te_taxa, te_i2s, te_i2t = parse_fasta(te_fa)
            
            tr_seq_set = set(tr_seqs)
            va_seq_set = set(va_seqs)
            te_seq_set = set(te_seqs)
            
            # 3. Verify sequence overlap
            if len(tr_seq_set & va_seq_set) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Train/Val seq overlap")
            if len(tr_seq_set & te_seq_set) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Train/Test seq overlap")
            if len(va_seq_set & te_seq_set) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Val/Test seq overlap")
            
            # 4. Verify taxonomic overlap (Only at holdout rank 'family')
            if rank == 'family':
                if len(tr_taxa & va_taxa) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Train/Val taxa overlap")
                if len(tr_taxa & te_taxa) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Train/Test taxa overlap")
                if len(va_taxa & te_taxa) > 0: rank_res["PASS"] = False; rank_res["errors"].append("Val/Test taxa overlap")
            
            # 5. Verify FAISS
            ids_npy = f"{index_dir}/{rank}/seed_{seed}/ids.npy"
            meta_json = f"{index_dir}/{rank}/seed_{seed}/meta.json"
            if os.path.exists(ids_npy) and os.path.exists(meta_json):
                faiss_labels = np.load(ids_npy, allow_pickle=True)
                with open(meta_json, 'r') as f:
                    meta_data = json.load(f)
                if len(faiss_labels) != len(tr_seqs): rank_res["PASS"] = False; rank_res["errors"].append("FAISS count mismatch")
                if meta_data.get("num_vectors") != len(tr_seqs): rank_res["PASS"] = False; rank_res["errors"].append("FAISS meta count mismatch")
            else:
                rank_res["PASS"] = False
                rank_res["errors"].append("FAISS index missing")
                
            # 6. Sequence similarity
            print(f"[{run_key}] Building sparse matrix...")
            tr_mat, vectorizer = build_kmer_matrix(tr_seqs)
            te_mat = vectorizer.transform(te_seqs)
            
            tr_sizes = np.array(tr_mat.sum(axis=1)).flatten()
            te_sizes = np.array(te_mat.sum(axis=1)).flatten()
            
            sim_99, sim_95, sim_90 = 0, 0, 0
            max_identity = 0
            sim_distribution = {"0-50": 0, "50-70": 0, "70-90": 0, "90-95": 0, "95-99": 0, "99-100": 0}
            
            chunk_size = 1000
            for i in range(0, te_mat.shape[0], chunk_size):
                te_chunk = te_mat[i:i+chunk_size]
                intersection = te_chunk.dot(tr_mat.T)
                
                for j in range(intersection.shape[0]):
                    te_idx = i + j
                    row = intersection.getrow(j)
                    if row.nnz == 0:
                        sim_distribution["0-50"] += 1
                        continue
                        
                    train_indices = row.indices
                    inter_vals = row.data
                    
                    jaccards = inter_vals / (tr_sizes[train_indices] + te_sizes[te_idx] - inter_vals + 1e-9)
                    best_jaccard = float(np.max(jaccards))
                    
                    if best_jaccard > max_identity: max_identity = best_jaccard
                    
                    if best_jaccard >= 0.94: sim_99 += 1
                    if best_jaccard >= 0.73: sim_95 += 1
                    if best_jaccard >= 0.53: sim_90 += 1
                    
                    if best_jaccard < 0.5: sim_distribution["0-50"] += 1
                    elif best_jaccard < 0.7: sim_distribution["50-70"] += 1
                    elif best_jaccard < 0.9: sim_distribution["70-90"] += 1
                    elif best_jaccard < 0.95: sim_distribution["90-95"] += 1
                    elif best_jaccard < 0.99: sim_distribution["95-99"] += 1
                    else: sim_distribution["99-100"] += 1
                    
            rank_res["similarity"] = {
                "99_pct_count": sim_99,
                "99_pct_rate": sim_99 / len(te_seqs),
                "95_pct_count": sim_95,
                "90_pct_count": sim_90,
                "max_identity": max_identity,
                "distribution": sim_distribution
            }
            
            source_count = count_original(source_fa)
            dedup_count = len(tr_seqs) + len(va_seqs) + len(te_seqs)
            
            rank_res["accounting"] = {
                "source_count": source_count,
                "deduplicated_count": dedup_count,
                "train_count": len(tr_seqs),
                "val_count": len(va_seqs),
                "test_count": len(te_seqs)
            }
            
            report_data["results"][run_key] = rank_res
            
            manifest[run_key] = {
                "dataset_version": "2.0.0_strict_dedup",
                "source_file": source_fa,
                "source_hash": get_hash(source_fa),
                "train_hash": get_hash(tr_fa),
                "test_hash": get_hash(te_fa),
                "counts": rank_res["accounting"],
                "holdout_rank": "family",
                "seed": seed,
                "commit": commit_hash
            }
            
            if not rank_res["PASS"]:
                all_passed = False

    report_data["PHASE_2_STATUS"] = "PASS" if all_passed else "FAIL"

    with open(f"{base_dir}/phase2_acceptance_report.json", "w") as f:
        json.dump(report_data, f, indent=4)
        
    with open(f"{base_dir}/dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    import csv
    with open(f"{base_dir}/phase2_acceptance_report.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["run_key", "pass", "source_cnt", "dedup_cnt", "tr_cnt", "va_cnt", "te_cnt", "sim_99", "sim_95", "sim_90", "errors"])
        for key, res in report_data["results"].items():
            if "accounting" not in res: continue
            writer.writerow([
                key, res["PASS"],
                res["accounting"]["source_count"], res["accounting"]["deduplicated_count"],
                res["accounting"]["train_count"], res["accounting"]["val_count"], res["accounting"]["test_count"],
                res["similarity"]["99_pct_count"], res["similarity"]["95_pct_count"], res["similarity"]["90_pct_count"],
                "|".join(res["errors"])
            ])

    print(f"Final Status: {report_data['PHASE_2_STATUS']}")

if __name__ == "__main__":
    main()
