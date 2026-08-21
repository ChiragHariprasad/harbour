import os
import sys
import json
import csv
import time
import argparse
import traceback
import subprocess
import numpy as np
import concurrent.futures
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
)

BASE_DIR = "E:/HTRP"
OUTPUT_BASE = "E:/paper_revamp/phase15_vsearch_baseline"
RANKS = ["class", "order", "family", "genus"]
SEEDS = [42, 43, 44, 45, 46]
MAX_WORKERS = 2
VSEARCH_BIN = "E:/HTRP/vsearch/vsearch-2.28.1-win-x86_64/bin/vsearch.exe"

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
                tax_map[parts[source_idx]] = parts[target_idx]
    return tax_map

def run_vsearch(query_fa, db_fa, out_tsv):
    cmd = [
        VSEARCH_BIN,
        "--usearch_global", query_fa,
        "--db", db_fa,
        "--id", "0.0",
        "--maxaccepts", "1",
        "--blast6out", out_tsv,
        "--threads", "16"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def parse_blast6out(tsv_file):
    res = {}
    if not os.path.exists(tsv_file):
        return res
    with open(tsv_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qseqid = parts[0]
                sseqid = parts[1]
                pident = float(parts[2])
                res[qseqid] = (sseqid, pident)
    return res

def eval_single_config(rank, seed):
    try:
        t0 = time.time()
        cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
        output_dir = f"{OUTPUT_BASE}/{rank}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)

        tax_map = load_taxonomy_map(target_rank=rank)
        
        # 1. Build Reference DB from train_fit.fasta
        fit_fa = f"{cache_dir}/train_fit.fasta"
        db_fa = f"{output_dir}/vsearch_db.fasta"
        
        fit_records = list(SeqIO.parse(fit_fa, "fasta"))
        fit_taxa = set()
        
        db_records = []
        for r in fit_records:
            acc = r.id.split('|')[0]
            target_label = tax_map.get(acc, "Unknown")
            fit_taxa.add(target_label)
            r.id = f"{acc}|{target_label}"
            r.description = ""
            db_records.append(r)
            
        SeqIO.write(db_records, db_fa, "fasta")
        
        # Spot check for constraint 9
        if rank == "class" and seed == 42:
            print("--- SPOT CHECK (Constraint 9) ---")
            print(f"Rank: {rank}")
            for r in db_records[:3]:
                print(f"Header: {r.id}")
            print("---------------------------------")
            
        # 2. Build Validation Query (train_eval + val)
        eval_fa = f"{cache_dir}/train_eval.fasta"
        val_fa = f"{cache_dir}/val.fasta"
        
        eval_records = list(SeqIO.parse(eval_fa, "fasta"))
        val_records = list(SeqIO.parse(val_fa, "fasta"))
        
        # Enforce disjointness
        eval_records = [r for r in eval_records if tax_map.get(r.id.split('|')[0], "Unknown") in fit_taxa]
        val_records = [r for r in val_records if tax_map.get(r.id.split('|')[0], "Unknown") not in fit_taxa]
        
        calib_query_fa = f"{output_dir}/calib_query.fasta"
        SeqIO.write(eval_records + val_records, calib_query_fa, "fasta")
        
        # 3. Build Test Query (train_eval + test)
        test_fa = f"{cache_dir}/test.fasta"
        test_records = list(SeqIO.parse(test_fa, "fasta"))
        test_records = [r for r in test_records if tax_map.get(r.id.split('|')[0], "Unknown") not in fit_taxa]
        
        test_query_fa = f"{output_dir}/test_query.fasta"
        final_test_records = eval_records + test_records
        SeqIO.write(final_test_records, test_query_fa, "fasta")
        
        # 4. Run VSEARCH
        calib_out_tsv = f"{output_dir}/calib_out.tsv"
        test_out_tsv = f"{output_dir}/test_out.tsv"
        
        run_vsearch(calib_query_fa, db_fa, calib_out_tsv)
        run_vsearch(test_query_fa, db_fa, test_out_tsv)
        
        calib_hits = parse_blast6out(calib_out_tsv)
        
        def get_preds(records, hits_dict):
            preds = []
            for r in records:
                acc = r.id.split('|')[0]
                true_taxon = tax_map.get(acc, "Unknown")
                true_status = "KNOWN" if true_taxon in fit_taxa else "NOVEL"
                
                search_id = r.id
                if search_id in hits_dict:
                    sseqid, pident = hits_dict[search_id]
                    predicted_taxon = sseqid.split('|')[-1]
                else:
                    predicted_taxon = "Unknown"
                    pident = 0.0
                    
                preds.append({
                    "sequence_id": acc,
                    "true_taxon": true_taxon,
                    "true_status": true_status,
                    "predicted_taxon": predicted_taxon,
                    "pident": pident
                })
            return preds
            
        calib_preds = get_preds(eval_records + val_records, calib_hits)
        
        # 5. Calibrate Threshold
        known_total = sum(1 for p in calib_preds if p["true_status"] == "KNOWN")
        native_correct = sum(1 for p in calib_preds if p["true_status"] == "KNOWN" and p["predicted_taxon"] == p["true_taxon"])
        a_cs = native_correct / known_total if known_total > 0 else 0
        constraint_target = 0.8 * a_cs
        
        novel_total = sum(1 for p in calib_preds if p["true_status"] == "NOVEL")
        
        best_threshold = 0.0
        best_h_score = 0
        
        # pident is 0-100
        for threshold in np.arange(0.0, 100.1, 0.5):
            known_correct_os = sum(1 for p in calib_preds if p["true_status"] == "KNOWN" and p["pident"] >= threshold and p["predicted_taxon"] == p["true_taxon"])
            novel_correct = sum(1 for p in calib_preds if p["true_status"] == "NOVEL" and p["pident"] < threshold)
            
            a_known_os = known_correct_os / known_total if known_total > 0 else 0
            r_novel = novel_correct / novel_total if novel_total > 0 else 0
            
            h_score = 2 * a_known_os * r_novel / (a_known_os + r_novel) if (a_known_os + r_novel) > 0 else 0
            
            if a_known_os >= constraint_target and h_score > best_h_score:
                best_h_score = h_score
                best_threshold = threshold
                
        # 6. Evaluate Test
        test_hits = parse_blast6out(test_out_tsv)
        test_preds = get_preds(final_test_records, test_hits)
        
        rows = []
        for p in test_preds:
            predicted_status = "KNOWN" if p["pident"] >= best_threshold else "NOVEL"
            rows.append({
                "experiment_id": f"{rank}_{seed}",
                "rank": rank,
                "seed": seed,
                "sequence_id": p["sequence_id"],
                "true_taxon": p["true_taxon"],
                "true_status": p["true_status"],
                "predicted_taxon": p["predicted_taxon"],
                "predicted_status": predicted_status,
                "pident": p["pident"],
                "threshold_pident": best_threshold,
                "native_a_cs": a_cs
            })
            
        pred_file = os.path.join(output_dir, "predictions.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            
        # Metrics
        y_true_status = [r["true_status"] for r in rows]
        y_pred_status = [r["predicted_status"] for r in rows]
        y_true_taxon = [r["true_taxon"] for r in rows]
        y_pred_taxon = [r["predicted_taxon"] for r in rows]
        pidents = [r["pident"] / 100.0 for r in rows]
        
        known_indices = [i for i, s in enumerate(y_true_status) if s == "KNOWN"]
        novel_indices = [i for i, s in enumerate(y_true_status) if s == "NOVEL"]

        if known_indices:
            known_correct = sum(1 for i in known_indices if y_pred_status[i] == "KNOWN" and y_pred_taxon[i] == y_true_taxon[i])
            a_known = known_correct / len(known_indices)
        else:
            a_known = float('nan')

        if novel_indices:
            novel_correct = sum(1 for i in novel_indices if y_pred_status[i] == "NOVEL")
            r_novel = novel_correct / len(novel_indices)
        else:
            r_novel = float('nan')

        if a_known > 0 and r_novel > 0 and not (np.isnan(a_known) or np.isnan(r_novel)):
            h_score = 2 * a_known * r_novel / (a_known + r_novel)
        else:
            h_score = float('nan')

        binary_true = [1 if s == "KNOWN" else 0 for s in y_true_status]
        try:
            auroc = roc_auc_score(binary_true, pidents)
            auprc = average_precision_score(binary_true, pidents)
        except:
            auroc = float('nan')
            auprc = float('nan')

        if known_indices:
            yt_known = [y_true_taxon[i] for i in known_indices]
            yp_known = [y_pred_taxon[i] if y_pred_status[i] == "KNOWN" else "__REJECTED__" for i in known_indices]
            closed_acc = accuracy_score(yt_known, yp_known)
            closed_prec = precision_score(yt_known, yp_known, average='macro', zero_division=0)
            closed_rec = recall_score(yt_known, yp_known, average='macro', zero_division=0)
            closed_f1 = f1_score(yt_known, yp_known, average='macro', zero_division=0)
            closed_wf1 = f1_score(yt_known, yp_known, average='weighted', zero_division=0)
            closed_balacc = balanced_accuracy_score(yt_known, yp_known)
            closed_mcc = matthews_corrcoef(yt_known, yp_known)
        else:
            closed_acc = closed_prec = closed_rec = closed_f1 = closed_wf1 = closed_balacc = closed_mcc = float('nan')

        summary = {
            "rank": rank,
            "seed": seed,
            "threshold": best_threshold,
            "native_a_cs": a_cs,
            "A_known": a_known,
            "R_novel": r_novel,
            "H_Score": h_score,
            "AUROC": auroc,
            "AUPRC": auprc,
            "accuracy": closed_acc,
            "macro_precision": closed_prec,
            "macro_recall": closed_rec,
            "macro_f1": closed_f1,
            "weighted_f1": closed_wf1,
            "balanced_accuracy": closed_balacc,
            "MCC": closed_mcc,
            "eval_time_s": time.time() - t0
        }
        
        return {"status": "OK", **summary}
    except Exception as e:
        return {"rank": rank, "seed": seed, "status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}

def main():
    print("Starting Phase 15 VSEARCH Baseline Evaluation")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    configs = [(r, s) for r in RANKS for s in SEEDS]
    results = []
    t_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_config = {executor.submit(eval_single_config, r, s): (r, s) for r, s in configs}
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                res = future.result()
                results.append(res)
                print(f"[{config[0]}/{config[1]}] Completed. H={res.get('H_Score',0):.4f} A_known={res.get('A_known',0):.4f} R_novel={res.get('R_novel',0):.4f} tau={res.get('threshold',0):.2f}% Time={res.get('eval_time_s',0):.1f}s")
            except Exception as e:
                print(f"[{config[0]}/{config[1]}] ERROR: {e}")
                results.append({"rank": config[0], "seed": config[1], "status": "FAILED", "error": str(e)})

    total_time = time.time() - t_start
    print(f"Full evaluation completed in {total_time:.2f}s")

    ok_results = [r for r in results if r.get("status") == "OK"]
    
    if ok_results:
        summary_file = f"{OUTPUT_BASE}/phase15_vsearch_baseline_summary.csv"
        fieldnames = [k for k in ok_results[0].keys() if k not in ("status", "traceback", "error")]
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(ok_results, key=lambda x: (x["rank"], x["seed"])):
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        agg_rows = []
        metrics_to_agg = ["A_known", "R_novel", "H_Score", "AUROC", "AUPRC",
                          "accuracy", "macro_precision", "macro_recall", "macro_f1",
                          "weighted_f1", "balanced_accuracy", "MCC"]

        for rank in RANKS:
            rank_results = [r for r in ok_results if r["rank"] == rank]
            if not rank_results:
                continue
            agg = {"rank": rank, "n_seeds": len(rank_results)}
            for metric in metrics_to_agg:
                vals = [r[metric] for r in rank_results if not (isinstance(r[metric], float) and np.isnan(r[metric]))]
                if vals:
                    agg[f"{metric}_mean"] = np.mean(vals)
                    agg[f"{metric}_std"] = np.std(vals)
                else:
                    agg[f"{metric}_mean"] = float('nan')
                    agg[f"{metric}_std"] = float('nan')
            agg_rows.append(agg)

        agg_file = f"{OUTPUT_BASE}/vsearch_aggregate_metrics.csv"
        if agg_rows:
            with open(agg_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=agg_rows[0].keys())
                writer.writeheader()
                writer.writerows(agg_rows)

        print("\nAggregated Results:")
        for agg in agg_rows:
            print(f"{agg['rank']:<10} H={agg.get('H_Score_mean',0):.4f}+-{agg.get('H_Score_std',0):.4f} A_known={agg.get('A_known_mean',0):.4f} R_novel={agg.get('R_novel_mean',0):.4f}")
            
if __name__ == "__main__":
    main()
