import os
import sys
import time
import shutil
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import concurrent.futures

BASE_DIR = "E:/HTRP"
OUTPUT_BASE = "E:/paper_revamp/phase16_kraken2_baseline"
WSL_KRAKEN2_BUILD = "/home/chirag/kraken2_bin/kraken2-build"
WSL_KRAKEN2 = "/home/chirag/kraken2_bin/kraken2"

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

def build_db(rank, seed):
    print(f"--- Kraken2 Build: {rank} seed {seed} ---")
    t0 = time.time()
    cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    db_dir = f"{OUTPUT_BASE}/db_{rank}_{seed}"
    wsl_db_dir = db_dir.replace("E:/", "/mnt/e/")
    
    tax_map = load_taxonomy_map(target_rank=rank)
    fit_records = list(SeqIO.parse(f"{cache_dir}/train_fit.fasta", "fasta"))
    
    unique_labels = sorted(list(set(tax_map.get(r.id.split('|')[0], "Unknown") for r in fit_records)))
    label_to_taxid = {label: i+2 for i, label in enumerate(unique_labels)}
    label_to_taxid["Root"] = 1
    taxid_to_label = {v: k for k, v in label_to_taxid.items()}
    
    if os.path.exists(db_dir) and os.path.exists(f"{db_dir}/hash.k2d"):
        print(f"[{rank}/{seed}] DB already built, skipping.")
        return wsl_db_dir, label_to_taxid, taxid_to_label
    os.makedirs(f"{db_dir}/taxonomy", exist_ok=True)
    
    tax_map = load_taxonomy_map(target_rank=rank)
    fit_records = list(SeqIO.parse(f"{cache_dir}/train_fit.fasta", "fasta"))
    
    unique_labels = sorted(list(set(tax_map.get(r.id.split('|')[0], "Unknown") for r in fit_records)))
    label_to_taxid = {label: i+2 for i, label in enumerate(unique_labels)}
    label_to_taxid["Root"] = 1
    taxid_to_label = {v: k for k, v in label_to_taxid.items()}
    
    with open(f"{db_dir}/taxonomy/nodes.dmp", "w", newline='\n') as f:
        f.write("1\t|\t1\t|\tno rank\t|\n")
        for label, taxid in label_to_taxid.items():
            if taxid == 1: continue
            f.write(f"{taxid}\t|\t1\t|\t{rank}\t|\n")
            
    with open(f"{db_dir}/taxonomy/names.dmp", "w", newline='\n') as f:
        f.write("1\t|\tRoot\t|\t\t|\tscientific name\t|\n")
        for label, taxid in label_to_taxid.items():
            if taxid == 1: continue
            f.write(f"{taxid}\t|\t{label}\t|\t\t|\tscientific name\t|\n")
            
    seq_out = []
    for r in fit_records:
        acc = r.id.split('|')[0]
        label = tax_map.get(acc, "Unknown")
        taxid = label_to_taxid[label]
        r.id = f"kraken:taxid|{taxid}|{acc}|{label}"
        r.description = ""
        seq_out.append(r)
        
    temp_fa = f"{db_dir}/temp.fasta"
    wsl_temp_fa = wsl_db_dir + "/temp.fasta"
    SeqIO.write(seq_out, temp_fa, "fasta")
    
    subprocess.run(f"wsl bash -c '{WSL_KRAKEN2_BUILD} --add-to-library {wsl_temp_fa} --db {wsl_db_dir}'", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(temp_fa)
    
    subprocess.run(f"wsl bash -c '{WSL_KRAKEN2_BUILD} --build --db {wsl_db_dir} --threads 16'", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[{rank}/{seed}] Build completed in {time.time() - t0:.1f}s")
    
    return wsl_db_dir, label_to_taxid, taxid_to_label

def parse_kraken2_output(kraken_out_path, eval_records, label_to_taxid, taxid_to_label, closed_set_labels, tax_map):
    # Parse kraken standard output
    # C/U  seq_id  tax_id  length  kmer_string
    preds = []
    y_true = []
    accs = []
    
    with open(kraken_out_path, "r") as f:
        for line, r in zip(f, eval_records):
            parts = line.strip().split('\t')
            c_u = parts[0]
            acc = parts[1] # original ID from FASTA
            tax_id = int(parts[2])
            
            # Reconstruct original accession correctly if needed
            acc_clean = acc.split('|')[0]
            true_label = tax_map.get(acc_clean, "Unknown")
            
            # Extract confidence from kmer string if possible
            # But kraken 2 natively uses its own confidence, or we can just parse kmer string
            kmer_string = parts[4]
            kmer_parts = kmer_string.split()
            
            total_kmers = 0
            pred_kmers = 0
            for kp in kmer_parts:
                if '|' in kp:
                    kp = kp.split('|')[0] # handle ambiguous mappings
                if ':' not in kp: continue
                
                tid, count = kp.split(':')
                count = int(count)
                if tid != "A": # "A" means ambiguous
                    total_kmers += count
                    if int(tid) == tax_id:
                        pred_kmers += count
                        
            conf = pred_kmers / total_kmers if total_kmers > 0 else 0.0
            
            if c_u == 'U' or tax_id not in taxid_to_label:
                pred_label = "Unclassified"
                conf = 0.0
            else:
                pred_label = taxid_to_label[tax_id]
                
            preds.append((pred_label, conf))
            y_true.append(true_label)
            accs.append(acc_clean)
            
    return accs, y_true, preds

def compute_metrics(y_true, predictions, tau, closed_set_labels):
    TP = FP = TN = FN = correct_known = total_known = 0
    
    for true_lbl, (pred_lbl, conf) in zip(y_true, predictions):
        novel_pred = conf < tau or pred_lbl == "Unclassified"
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

def evaluate_config(rank, seed):
    wsl_db_dir, label_to_taxid, taxid_to_label = build_db(rank, seed)
    t0 = time.time()
    
    tax_map = load_taxonomy_map(rank)
    cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    out_dir = f"{OUTPUT_BASE}/db_{rank}_{seed}"
    
    fit_records = list(SeqIO.parse(f"{cache_dir}/train_fit.fasta", "fasta"))
    closed_set_labels = set([tax_map.get(r.id.split('|')[0], "Unknown") for r in fit_records])
    
    # Validation Eval
    val_records = list(SeqIO.parse(f"{cache_dir}/train_eval.fasta", "fasta")) + list(SeqIO.parse(f"{cache_dir}/val.fasta", "fasta"))
    val_fa = f"{out_dir}/val_combined.fasta"
    SeqIO.write(val_records, val_fa, "fasta")
    wsl_val_fa = val_fa.replace("E:/", "/mnt/e/")
    
    val_out = f"{out_dir}/val.kraken"
    cmd1 = f"wsl bash -c \"{WSL_KRAKEN2} --db {wsl_db_dir} --threads 16 {wsl_val_fa} > {val_out.replace('E:/', '/mnt/e/')}\""
    res1 = subprocess.run(cmd1, shell=True)
    if res1.returncode != 0: print(f"Warning: val.kraken returned {res1.returncode}")
    
    accs, val_true, val_preds = parse_kraken2_output(val_out, val_records, label_to_taxid, taxid_to_label, closed_set_labels, tax_map)
    
    # 1. Base Accuracy on Eval
    eval_len = len(list(SeqIO.parse(f"{cache_dir}/train_eval.fasta", "fasta")))
    eval_true = val_true[:eval_len]
    eval_preds = val_preds[:eval_len]
    correct_known_base = sum(1 for t, p in zip(eval_true, eval_preds) if t == p[0])
    A_cs = correct_known_base / len(eval_true) if len(eval_true) > 0 else 0
    
    # Sweep threshold on Validation
    best_H = -1; best_tau = 0.0
    for tau in np.arange(0.0, 1.05, 0.05):
        H, A_k, R_n = compute_metrics(val_true, val_preds, tau, closed_set_labels)
        if A_k >= 0.8 * A_cs and H > best_H:
            best_H = H
            best_tau = tau
            
    # Test Eval
    test_records = list(SeqIO.parse(f"{cache_dir}/train_eval.fasta", "fasta")) + list(SeqIO.parse(f"{cache_dir}/test.fasta", "fasta"))
    test_fa = f"{out_dir}/test_combined.fasta"
    SeqIO.write(test_records, test_fa, "fasta")
    wsl_test_fa = test_fa.replace("E:/", "/mnt/e/")
    
    test_out = f"{out_dir}/test.kraken"
    cmd2 = f"wsl bash -c \"{WSL_KRAKEN2} --db {wsl_db_dir} --threads 16 {wsl_test_fa} > {test_out.replace('E:/', '/mnt/e/')}\""
    res2 = subprocess.run(cmd2, shell=True)
    if res2.returncode != 0: print(f"Warning: test.kraken returned {res2.returncode}")
    
    test_accs, test_true, test_preds = parse_kraken2_output(test_out, test_records, label_to_taxid, taxid_to_label, closed_set_labels, tax_map)
    
    H, A_k, R_n = compute_metrics(test_true, test_preds, best_tau, closed_set_labels)
    
    df = pd.DataFrame({
        'accession': test_accs,
        'true_label': test_true,
        'pred_label': [p[0] for p in test_preds],
        'confidence': [p[1] for p in test_preds],
        'is_novel': [p[1] < best_tau or p[0] == "Unclassified" for p in test_preds]
    })
    df.to_csv(f"{out_dir}/predictions.csv", index=False)
    
    print(f"[{rank}/{seed}] Finished. H={H:.4f} A_known={A_k:.4f} R_novel={R_n:.4f} tau={best_tau}")
    return {
        'rank': rank, 'seed': seed, 
        'H_score': H, 'A_known': A_k, 'R_novel': R_n, 
        'tau': best_tau
    }

def main():
    ranks = ['class', 'order', 'family', 'genus']
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    for rank in ranks:
        for seed in seeds:
            results.append(evaluate_config(rank, seed))
            
    df = pd.DataFrame(results)
    print("\nAggregated Results:")
    for rank in ranks:
        sub = df[df['rank'] == rank]
        h_mean = sub['H_score'].mean()
        h_std = sub['H_score'].std()
        ak_mean = sub['A_known'].mean()
        rn_mean = sub['R_novel'].mean()
        print(f"{rank:10s} H={h_mean:.4f}+-{h_std:.4f} A_known={ak_mean:.4f} R_novel={rn_mean:.4f}")
        
    df.to_csv(f"{OUTPUT_BASE}/kraken2_aggregate_metrics.csv", index=False)

if __name__ == "__main__":
    main()
