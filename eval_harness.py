import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from hierarchical_infer import HierarchicalPipeline

def calc_fpr95(y_true, y_score):
    """Calculate FPR at 95% TPR (lower is better).
       y_true: 1 for NOVEL (positive class), 0 for KNOWN.
       y_score: anomaly score (higher means more likely NOVEL).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # Find the threshold where recall is at least 0.95
    idx = np.where(recalls >= 0.95)[0]
    if len(idx) > 0:
        idx = idx[-1]
    else:
        idx = 0
    thresh = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    
    # Calculate FPR at this threshold
    fp = np.sum((y_score >= thresh) & (y_true == 0))
    tn = np.sum((y_score < thresh) & (y_true == 0))
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)

def run_evaluation(args):
    print(f"Loading test data from {args.test_fasta}...")
    test_seqs = []
    y_true = [] # 1 if true novel, 0 if true known
    
    known_taxa = set()
    print(f"Reading train data to determine known taxa from {args.train_fasta}...")
    for rec in SeqIO.parse(args.train_fasta, "fasta"):
        parts = rec.id.split("|")
        label = parts[-1] if len(parts) > 1 else rec.id
        known_taxa.add(label)
        
    print(f"Total known taxa: {len(known_taxa)}")
    
    for rec in SeqIO.parse(args.test_fasta, "fasta"):
        parts = rec.id.split("|")
        label = parts[-1] if len(parts) > 1 else rec.id
        test_seqs.append(str(rec.seq))
        y_true.append(0 if label in known_taxa else 1)
        
    print(f"Total queries: {len(test_seqs)} ({sum(y_true)} NOVEL, {len(y_true) - sum(y_true)} KNOWN)")
    
    pipeline = HierarchicalPipeline(marker=args.marker)
    
    # Parse ablations
    use_jaccard = not args.no_jaccard
    use_novelty = not args.no_novelty
    use_hierarchy = not args.no_hierarchy
    
    print(f"Ablations: Jaccard={use_jaccard}, Novelty={use_novelty}, Hierarchy={use_hierarchy}")
    print("Running inference...")
    
    y_pred = []
    y_score = [] # Anomaly score
    times = []
    
    for i, seq in enumerate(test_seqs):
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(test_seqs)}")
            
        t0 = time.time()
        res = pipeline.predict(seq, use_jaccard=use_jaccard, use_novelty=use_novelty, use_hierarchy=use_hierarchy)
        t_elapsed = time.time() - t0
        
        pred_label = 1 if "NOVEL" in res["prediction"] else 0
        y_pred.append(pred_label)
        
        if "max_z" in res:
            score = res["max_z"]
        else:
            score = -5.0
            
        y_score.append(score)
        times.append(t_elapsed)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    avg_latency = sum(times) / len(times) * 1000
    
    # Metrics
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recalls, precisions)
        fpr95 = calc_fpr95(y_true, y_score)
    else:
        auroc, auprc, fpr95 = 0.0, 0.0, 0.0
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0) # Novelty recall
    acc = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print("\n===== METRICS =====")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Novelty Recall:  {recall:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"AUROC:           {auroc:.4f}")
    print(f"AUPRC:           {auprc:.4f}")
    print(f"FPR:             {fpr:.4f}")
    print(f"FPR@95TPR:       {fpr95:.4f}")
    print(f"Avg Latency:     {avg_latency:.2f} ms")
    
    out = {
        "ablations": {
            "jaccard": use_jaccard,
            "novelty": use_novelty,
            "hierarchy": use_hierarchy
        },
        "metrics": {
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "auroc": auroc,
            "auprc": auprc,
            "fpr": fpr,
            "fpr95": fpr95,
            "latency_ms": avg_latency
        },
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=str, default="16S")
    parser.add_argument("--train_fasta", type=str, required=True)
    parser.add_argument("--test_fasta", type=str, required=True)
    parser.add_argument("--out", type=str, default="eval_results.json")
    
    # Ablations
    parser.add_argument("--no_jaccard", action="store_true")
    parser.add_argument("--no_novelty", action="store_true")
    parser.add_argument("--no_hierarchy", action="store_true")
    
    args = parser.parse_args()
    run_evaluation(args)
