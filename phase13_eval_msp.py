import os
import sys
import json
import csv
import time
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import concurrent.futures
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import encode_sequence, DEVICE
from train_softmax_baseline import CNNClassifier

BASE_DIR = "E:/HTRP"
OUTPUT_BASE = "E:/paper_revamp/phase13_msp_baseline"
RANKS = ["class", "order", "family", "genus"]
SEEDS = [42, 43, 44, 45, 46]
MAX_WORKERS = 10

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

def load_model_and_labels(model_dir):
    with open(os.path.join(model_dir, "label_map.json"), "r") as f:
        label_info = json.load(f)

    num_classes = label_info["num_classes"]
    idx_to_label = {int(k): v for k, v in label_info["idx_to_label"].items()}
    label_to_idx = label_info["label_to_idx"]

    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, "classifier.pt"), map_location=DEVICE))
    model.eval()

    return model, idx_to_label, label_to_idx

def predict_sequences(model, sequences, idx_to_label):
    results = []
    model.eval()
    with torch.no_grad():
        for seq_str in sequences:
            x = encode_sequence(seq_str).to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            max_prob, pred_idx = probs.max(dim=1)

            predicted_label = idx_to_label.get(pred_idx.item(), "UNKNOWN")
            msp_score = max_prob.item()

            results.append({
                "predicted_taxon_raw": predicted_label,
                "msp_score": msp_score,
                "predicted_idx": pred_idx.item()
            })
    return results

def eval_single_config(rank, seed):
    try:
        t0 = time.time()
        cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
        model_dir = f"{BASE_DIR}/assets/HTRP_models/softmax_baseline/16S/{rank}/seed_{seed}"
        output_dir = f"{OUTPUT_BASE}/{rank}/seed_{seed}"
        os.makedirs(output_dir, exist_ok=True)

        tax_map = load_taxonomy_map(target_rank=rank)
        
        # Load fit_taxa to establish ground truth for KNOWN classes
        fit_fa = f"{cache_dir}/train_fit.fasta"
        fit_records = list(SeqIO.parse(fit_fa, "fasta"))
        fit_taxa = set(tax_map.get(r.id.split('|')[0], "Unknown") for r in fit_records)

        model, idx_to_label, label_to_idx = load_model_and_labels(model_dir)

        # ---------------------------------------------------------
        # CALIBRATION
        # Uses train_eval.fasta (KNOWN) and val.fasta (NOVEL)
        # ---------------------------------------------------------
        val_fa = f"{cache_dir}/val.fasta"
        eval_fa = f"{cache_dir}/train_eval.fasta"
        
        val_records = list(SeqIO.parse(val_fa, "fasta"))
        eval_records = list(SeqIO.parse(eval_fa, "fasta"))
        
        # Filter to ensure strict disjointness
        eval_records = [r for r in eval_records if tax_map.get(r.id.split('|')[0], "Unknown") in fit_taxa]
        val_records = [r for r in val_records if tax_map.get(r.id.split('|')[0], "Unknown") not in fit_taxa]
        
        calib_records = eval_records + val_records
        
        def get_preds(records):
            sequences = [str(r.seq).upper() for r in records]
            seq_ids = [r.id.split('|')[0] for r in records]
            raw_preds = predict_sequences(model, sequences, idx_to_label)
            
            for i, p in enumerate(raw_preds):
                p["sequence_id"] = seq_ids[i]
                p["true_taxon"] = tax_map.get(seq_ids[i], "Unknown")
                p["true_status"] = "KNOWN" if p["true_taxon"] in fit_taxa else "NOVEL"
                # Translate from Family (which model learned) to Target Rank
                p["predicted_taxon"] = tax_map.get(p["predicted_taxon_raw"], p["predicted_taxon_raw"])
            return raw_preds

        calib_preds = get_preds(calib_records)

        # Calculate native closed-set accuracy (A_cs)
        known_total = sum(1 for p in calib_preds if p["true_status"] == "KNOWN")
        native_correct = sum(1 for p in calib_preds if p["true_status"] == "KNOWN" and p["predicted_taxon"] == p["true_taxon"])
        a_cs = native_correct / known_total if known_total > 0 else 0
        constraint_target = 0.8 * a_cs

        best_threshold = 0.5
        best_h_score = 0
        valid_threshold_found = False

        novel_total = sum(1 for p in calib_preds if p["true_status"] == "NOVEL")

        for threshold in np.arange(0.01, 1.0, 0.01):
            known_correct_os = sum(1 for p in calib_preds if p["true_status"] == "KNOWN" and p["msp_score"] >= threshold and p["predicted_taxon"] == p["true_taxon"])
            novel_correct = sum(1 for p in calib_preds if p["true_status"] == "NOVEL" and p["msp_score"] < threshold)
            
            a_known_os = known_correct_os / known_total if known_total > 0 else 0
            r_novel = novel_correct / novel_total if novel_total > 0 else 0

            h_score = 2 * a_known_os * r_novel / (a_known_os + r_novel) if (a_known_os + r_novel) > 0 else 0

            if a_known_os >= constraint_target:
                valid_threshold_found = True
                if h_score > best_h_score:
                    best_h_score = h_score
                    best_threshold = threshold

        if not valid_threshold_found:
            print(f"  [{rank}/{seed}] WARNING: No threshold met the 80% A_cs constraint (A_cs={a_cs:.4f}). Falling back to 0.5.")
            best_threshold = 0.5

        # ---------------------------------------------------------
        # FINAL EVALUATION
        # Uses train_eval.fasta (KNOWN) and test.fasta (NOVEL)
        # exactly like run_benchmark_phase6.py
        # ---------------------------------------------------------
        test_fa = f"{cache_dir}/test.fasta"
        test_records = list(SeqIO.parse(test_fa, "fasta"))
        test_records = [r for r in test_records if tax_map.get(r.id.split('|')[0], "Unknown") not in fit_taxa]
        
        final_eval_records = eval_records + test_records
        preds = get_preds(final_eval_records)

        rows = []
        for p in preds:
            predicted_status = "KNOWN" if p["msp_score"] >= best_threshold else "NOVEL"
            rows.append({
                "experiment_id": f"{rank}_{seed}",
                "rank": rank,
                "seed": seed,
                "sequence_id": p["sequence_id"],
                "true_taxon": p["true_taxon"],
                "true_status": p["true_status"],
                "predicted_taxon_raw": p["predicted_taxon_raw"],
                "predicted_taxon": p["predicted_taxon"],
                "predicted_status": predicted_status,
                "msp_score": p["msp_score"],
                "msp_threshold": best_threshold,
                "native_a_cs": a_cs
            })

        pred_file = os.path.join(output_dir, "predictions.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # ---------------------------------------------------------
        # METRICS COMPUTATION
        # ---------------------------------------------------------
        y_true_status = [r["true_status"] for r in rows]
        y_pred_status = [r["predicted_status"] for r in rows]
        y_true_taxon = [r["true_taxon"] for r in rows]
        y_pred_taxon = [r["predicted_taxon"] for r in rows]
        msp_scores = [r["msp_score"] for r in rows]

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
            auroc = roc_auc_score(binary_true, msp_scores)
            auprc = average_precision_score(binary_true, msp_scores)
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

        fk = sum(1 for r in rows if r["true_status"] == "NOVEL" and r["predicted_status"] == "KNOWN")
        fn = sum(1 for r in rows if r["true_status"] == "KNOWN" and r["predicted_status"] == "NOVEL")
        wrong_taxon = sum(1 for i in known_indices if y_pred_status[i] == "KNOWN" and y_pred_taxon[i] != y_true_taxon[i])

        fk_rate = fk / len(novel_indices) if novel_indices else float('nan')
        fn_rate = fn / len(known_indices) if known_indices else float('nan')
        wt_den = sum(1 for i in known_indices if y_pred_status[i] == "KNOWN")
        wrong_taxon_rate = wrong_taxon / wt_den if wt_den > 0 else float('nan')

        summary = {
            "rank": rank,
            "seed": seed,
            "msp_threshold": best_threshold,
            "native_a_cs": a_cs,
            "n_test": len(final_eval_records),
            "n_known": len(known_indices),
            "n_novel": len(novel_indices),
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
            "FK_count": fk,
            "FN_count": fn,
            "FK_rate": fk_rate,
            "FN_rate": fn_rate,
            "wrong_taxon_count": wrong_taxon,
            "wrong_taxon_rate": wrong_taxon_rate,
            "eval_time_s": time.time() - t0
        }

        with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return {"status": "OK", **summary}
    except Exception as e:
        return {"rank": rank, "seed": seed, "status": "FAILED", "error": str(e), "traceback": traceback.format_exc()}


def main():
    print(f"Starting Phase 13 Evaluation: MSP Baseline on Full Test Sets (with fixes!)")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    configs = [(r, s) for r in RANKS for s in SEEDS]
    results = []

    t_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_config = {executor.submit(eval_single_config, r, s): (r, s) for r, s in configs}
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{config[0]}/{config[1]}] Completed. H={result.get('H_Score',0):.4f} A_cs={result.get('native_a_cs',0):.4f} A_known={result.get('A_known',0):.4f} tau={result.get('msp_threshold',0):.2f}")
            except Exception as e:
                print(f"[{config[0]}/{config[1]}] ERROR: {e}")
                results.append({"rank": config[0], "seed": config[1], "status": "FAILED", "error": str(e)})

    ok_results = [r for r in results if r.get("status") == "OK"]
    
    if ok_results:
        summary_file = f"{OUTPUT_BASE}/phase13_msp_baseline_summary.csv"
        fieldnames = [k for k in ok_results[0].keys() if k not in ("status", "traceback", "error")]
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(ok_results, key=lambda x: (x["rank"], x["seed"])):
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        agg_rows = []
        metrics_to_agg = ["A_known", "R_novel", "H_Score", "AUROC", "AUPRC",
                          "accuracy", "macro_precision", "macro_recall", "macro_f1",
                          "weighted_f1", "balanced_accuracy", "MCC",
                          "FK_rate", "FN_rate", "wrong_taxon_rate"]

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

        agg_file = f"{OUTPUT_BASE}/msp_aggregate_metrics.csv"
        if agg_rows:
            with open(agg_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=agg_rows[0].keys())
                writer.writeheader()
                writer.writerows(agg_rows)

        print("\nAggregated Results:")
        for agg in agg_rows:
            print(f"{agg['rank']:<10} H={agg.get('H_Score_mean',0):.4f}+-{agg.get('H_Score_std',0):.4f} A_known={agg.get('A_known_mean',0):.4f} R_novel={agg.get('R_novel_mean',0):.4f}")
            
    else:
        print("All evaluations failed.")

if __name__ == "__main__":
    main()
