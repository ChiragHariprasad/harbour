"""
Softmax-CNN Baseline: Evaluation Script
Evaluates a trained softmax-CNN classifier for OOD detection using MSP thresholding.
Produces per-sequence predictions and summary metrics.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from Bio import SeqIO
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import encode_sequence, VOCAB_SIZE, DEVICE
from train_softmax_baseline import CNNClassifier


def load_model_and_labels(model_dir):
    """Load trained classifier and label map."""
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
    """Run inference on a list of sequences, returning predictions and MSP scores."""
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
                "predicted_taxon": predicted_label,
                "msp_score": msp_score,
                "predicted_idx": pred_idx.item()
            })
    return results


def calibrate_threshold(model, val_fasta, idx_to_label, label_to_idx, tax_map, target_rank):
    """Calibrate MSP threshold on validation set using the same holdout logic as SupCon."""
    sequences = []
    true_labels = []
    true_statuses = []

    for rec in SeqIO.parse(val_fasta, "fasta"):
        acc = rec.id.split("|")[0]
        true_taxon = tax_map.get(acc, "UNKNOWN")
        is_known = true_taxon in label_to_idx
        sequences.append(str(rec.seq))
        true_labels.append(true_taxon)
        true_statuses.append("KNOWN" if is_known else "NOVEL")

    preds = predict_sequences(model, sequences, idx_to_label)

    # Sweep MSP thresholds
    best_threshold = 0.5
    best_h_score = 0

    for threshold in np.arange(0.1, 0.99, 0.05):
        known_correct = 0
        known_total = 0
        novel_correct = 0
        novel_total = 0

        for i, p in enumerate(preds):
            if true_statuses[i] == "KNOWN":
                known_total += 1
                if p["msp_score"] >= threshold and p["predicted_taxon"] == true_labels[i]:
                    known_correct += 1
            else:
                novel_total += 1
                if p["msp_score"] < threshold:
                    novel_correct += 1

        a_known = known_correct / known_total if known_total > 0 else 0
        r_novel = novel_correct / novel_total if novel_total > 0 else 0

        if a_known + r_novel > 0:
            h_score = 2 * a_known * r_novel / (a_known + r_novel)
        else:
            h_score = 0

        if h_score > best_h_score:
            best_h_score = h_score
            best_threshold = threshold

    return best_threshold


def eval_single_config(rank, seed, base_dir):
    """Evaluate a single softmax-CNN config and produce metrics."""
    cache_dir = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    model_dir = f"{base_dir}/assets/HTRP_models/softmax_baseline/16S/{rank}/seed_{seed}"
    output_dir = f"{base_dir}/results/softmax_baseline/{rank}/seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Load taxonomy map
    tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    rank_idx = rank_map[rank]

    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > rank_idx:
                tax_map[parts[0]] = parts[rank_idx]

    # Load model
    model, idx_to_label, label_to_idx = load_model_and_labels(model_dir)

    # Calibrate threshold on validation set
    val_fasta = f"{cache_dir}/val.fasta"
    if os.path.exists(val_fasta):
        msp_threshold = calibrate_threshold(model, val_fasta, idx_to_label, label_to_idx, tax_map, rank)
    else:
        msp_threshold = 0.5
    print(f"[{rank}/{seed}] Calibrated MSP threshold: {msp_threshold:.2f}")

    # Evaluate on test set
    test_fasta = f"{cache_dir}/train_eval.fasta"
    if not os.path.exists(test_fasta):
        test_fasta = f"{cache_dir}/test.fasta"

    sequences = []
    seq_ids = []
    true_labels_list = []
    true_statuses = []

    for rec in SeqIO.parse(test_fasta, "fasta"):
        acc = rec.id.split("|")[0]
        true_taxon = tax_map.get(acc, "UNKNOWN")
        is_known = true_taxon in label_to_idx
        sequences.append(str(rec.seq))
        seq_ids.append(rec.id)
        true_labels_list.append(true_taxon)
        true_statuses.append("KNOWN" if is_known else "NOVEL")

    print(f"[{rank}/{seed}] Evaluating {len(sequences)} test sequences...")
    preds = predict_sequences(model, sequences, idx_to_label)

    # Build per-sequence results
    rows = []
    for i in range(len(sequences)):
        predicted_status = "KNOWN" if preds[i]["msp_score"] >= msp_threshold else "NOVEL"
        rows.append({
            "experiment_id": f"{rank}_{seed}",
            "rank": rank,
            "seed": seed,
            "sequence_id": seq_ids[i],
            "true_taxon": true_labels_list[i],
            "true_status": true_statuses[i],
            "predicted_taxon": preds[i]["predicted_taxon"],
            "predicted_status": predicted_status,
            "msp_score": preds[i]["msp_score"],
            "msp_threshold": msp_threshold
        })

    # Save per-sequence predictions
    import csv
    pred_file = os.path.join(output_dir, "predictions.csv")
    with open(pred_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ---- Compute metrics ----
    y_true_status = [r["true_status"] for r in rows]
    y_pred_status = [r["predicted_status"] for r in rows]
    y_true_taxon = [r["true_taxon"] for r in rows]
    y_pred_taxon = [r["predicted_taxon"] for r in rows]
    msp_scores = [r["msp_score"] for r in rows]

    # Known-class accuracy (among truly known, predicted known, is the taxon correct?)
    known_mask = [s == "KNOWN" for s in y_true_status]
    novel_mask = [s == "NOVEL" for s in y_true_status]

    known_indices = [i for i, m in enumerate(known_mask) if m]
    novel_indices = [i for i, m in enumerate(novel_mask) if m]

    # A_known: among known sequences, fraction predicted KNOWN with correct taxon
    if known_indices:
        known_correct = sum(1 for i in known_indices if y_pred_status[i] == "KNOWN" and y_pred_taxon[i] == y_true_taxon[i])
        a_known = known_correct / len(known_indices)
    else:
        a_known = float('nan')

    # R_novel: among novel sequences, fraction predicted NOVEL
    if novel_indices:
        novel_correct = sum(1 for i in novel_indices if y_pred_status[i] == "NOVEL")
        r_novel = novel_correct / len(novel_indices)
    else:
        r_novel = float('nan')

    # H-Score
    if a_known > 0 and r_novel > 0 and not (np.isnan(a_known) or np.isnan(r_novel)):
        h_score = 2 * a_known * r_novel / (a_known + r_novel)
    else:
        h_score = float('nan')

    # AUROC and AUPRC (binary: KNOWN=1, NOVEL=0; score = msp_score)
    binary_true = [1 if s == "KNOWN" else 0 for s in y_true_status]
    try:
        auroc = roc_auc_score(binary_true, msp_scores)
        auprc = average_precision_score(binary_true, msp_scores)
    except:
        auroc = float('nan')
        auprc = float('nan')

    # Closed-set metrics (on known sequences only, taxon-level)
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

    # Error rates
    total = len(rows)
    fk = sum(1 for r in rows if r["true_status"] == "NOVEL" and r["predicted_status"] == "KNOWN")
    fn = sum(1 for r in rows if r["true_status"] == "KNOWN" and r["predicted_status"] == "NOVEL")
    wrong_taxon = sum(1 for i in known_indices if y_pred_status[i] == "KNOWN" and y_pred_taxon[i] != y_true_taxon[i])

    fk_rate = fk / len(novel_indices) if novel_indices else float('nan')
    fn_rate = fn / len(known_indices) if known_indices else float('nan')
    wrong_taxon_rate = wrong_taxon / sum(1 for i in known_indices if y_pred_status[i] == "KNOWN") if sum(1 for i in known_indices if y_pred_status[i] == "KNOWN") > 0 else float('nan')

    summary = {
        "rank": rank,
        "seed": seed,
        "msp_threshold": msp_threshold,
        "n_test": total,
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
        "wrong_taxon_rate": wrong_taxon_rate
    }

    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{rank}/{seed}] H={h_score:.4f} AUROC={auroc:.4f} Acc={closed_acc:.4f} MCC={closed_mcc:.4f}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base_dir", type=str, default="E:/HTRP")
    args = parser.parse_args()

    result = eval_single_config(args.rank, args.seed, args.base_dir)
    print(json.dumps(result, default=str))
