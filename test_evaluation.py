import sys
import json
import random
import time
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, confusion_matrix
)
from src.baseline import ClassicalBaseline
from app import infer_ml, get_model, encode_sequence, DEVICE, TOP_K, get_index, Z_NOVELTY_THRESHOLD
import torch
# where the exact sequence is excluded from FAISS to prevent trivial 100% matches.
def infer_ml_loocv(seq: str, marker: str, seq_id: str):
    model = get_model(marker)
    X = encode_sequence(seq)
    with torch.no_grad():
        emb = model(X).cpu().numpy()
    # We must load the specific seeded FAISS index, NOT the production index via get_index
    seed = 42
    rank = "family"
    results = {}
    import faiss
    import json
    index_path = f"assets/HTRP_indexes/{marker}/{rank}/seed_{seed}/index.faiss"
    ids_path = f"assets/HTRP_indexes/{marker}/{rank}/seed_{seed}/ids.npy"
    meta_path = f"assets/HTRP_indexes/{marker}/{rank}/seed_{seed}/meta.json"
    
    index = faiss.read_index(index_path)
    ids = np.load(ids_path, allow_pickle=True)
    with open(meta_path) as f:
        stats = json.load(f)
        
    faiss.normalize_L2(emb)
    # Search for TOP_K + 1 to allow skipping the exact match
    D, I = index.search(emb, TOP_K + 5)
    
    hits_raw = ids[I[0]].tolist()
    dists_raw = D[0].tolist()
    
    hits = []
    dists = []
    for h, d in zip(hits_raw, dists_raw):
        # Leave one out (distance ~ 0 is the sequence itself)
        if d < 1e-6:
            continue
        hits.append(h)
        dists.append(d)
        if len(hits) == TOP_K:
            break
            
    if not hits: # Fallback if all top 5 were somehow exact matches
        hits = hits_raw[:TOP_K]
        dists = dists_raw[:TOP_K]
        
    import collections
    label, freq = collections.Counter(hits).most_common(1)[0]
    conf = freq / TOP_K
    
    # Use dists[0] to match the empirical stats calculation
    mean_dist = float(dists[0])
    z = (mean_dist - stats["mean"]) / stats["std"]
    
    prediction = label if conf >= 0.6 and z < Z_NOVELTY_THRESHOLD else "NOVEL"
    
    results[rank] = {
        "prediction": prediction,
        "confidence": round(conf, 4),
        "mean_distance": round(mean_dist, 4),
        "z_score": round(z, 3)
    }
    return results

def main():
    import torch
    
    seed = 42
    marker = "16S"
    rank = "family"
    
    train_fasta = f"assets/HTRP_cache/{marker}/{rank}/seed_{seed}/train.fasta"
    val_fasta = f"assets/HTRP_cache/{marker}/{rank}/seed_{seed}/val.fasta"
    
    # 1. Extract Training Taxa
    print("Extracting training taxa...")
    training_taxa = set()
    for rec in SeqIO.parse(train_fasta, "fasta"):
        parts = rec.id.split("|")
        training_taxa.add(parts[-1] if len(parts) > 1 else rec.id)
    print(f"Found {len(training_taxa)} training families.")
    
    # 2. Sample Evaluation Sequences
    print("Sampling Evaluation subsets...")
    random.seed(42)
    # Get Known Samples from Train (we'll use LOOCV in inference)
    known_recs = list(SeqIO.parse(train_fasta, "fasta"))
    random.shuffle(known_recs)
    known_samples = known_recs[:500]
    
    # Get Novel Samples from Val
    novel_recs = list(SeqIO.parse(val_fasta, "fasta"))
    random.shuffle(novel_recs)
    novel_samples = novel_recs[:500]
    
    mixed_eval_set = known_samples + novel_samples
    random.shuffle(mixed_eval_set)
    print(f"Constructed mixed evaluation set: {len(known_samples)} known + {len(novel_samples)} novel")
    
    # 3. Train Baseline
    baseline = ClassicalBaseline(k=4)
    baseline.train(train_fasta)
    
    # 4. Evaluate Mixed Set
    print("Evaluating Mixed Set (SupCon & Baseline)...")
    records = []
    
    # Pre-compute baseline predictions
    baseline_seqs = [str(r.seq) for r in mixed_eval_set]
    baseline_preds = baseline.predict_with_novelty(baseline_seqs, novelty_threshold=0.5)
    
    for i, rec in enumerate(mixed_eval_set):
        seq = str(rec.seq).upper()
        parts = rec.id.split("|")
        true_taxon = parts[-1] if len(parts) > 1 else rec.id
        true_novel = 1 if true_taxon not in training_taxa else 0
        
        # SupCon Prediction (using LOOCV for known samples)
        if true_novel == 0:
            res = infer_ml_loocv(seq, marker, rec.id)
        else:
            # We can't use infer_ml because it uses production index!
            # Use infer_ml_loocv but without the actual loocv (it won't match anyway since it's novel)
            res = infer_ml_loocv(seq, marker, rec.id)
            
        sup_fam = res.get(rank, {})
        sup_pred = sup_fam.get("prediction", "NOVEL")
        sup_z = sup_fam.get("z_score", 0.0)
        sup_predicted_novel = 1 if sup_pred == "NOVEL" else 0
        
        # We don't overwrite true_taxon!
        sup_predicted_taxon = sup_pred if sup_pred != "NOVEL" else ""
        
        # Baseline
        base_info = baseline_preds[i]
        base_pred_taxon = base_info["predicted_taxon"]
        base_pred_novel = base_info["predicted_novel"]
        base_nov_score = base_info["novelty_score"]
        
        records.append({
            "sequence_id": rec.id,
            "true_taxon": true_taxon,
            "true_novel": true_novel,
            "supcon_predicted_taxon": sup_pred,
            "supcon_predicted_novel": sup_predicted_novel,
            "supcon_novelty_score": sup_z,
            "baseline_predicted_taxon": base_pred_taxon,
            "baseline_predicted_novel": base_pred_novel,
            "baseline_novelty_score": base_nov_score
        })
        
    df = pd.DataFrame(records)
    df.to_csv("prediction_audit.csv", index=False)
    
    # 5. Closed-Set Metrics
    closed_df = df[df["true_novel"] == 0]
    
    def calc_closed(preds, y_true):
        # For closed-set, we expect the model to predict the known taxon.
        # If model predicts "NOVEL", it is a misclassification of a known taxon.
        return {
            "accuracy": accuracy_score(y_true, preds),
            "macro_precision": precision_score(y_true, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, preds, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, preds)
        }
        
    closed_res = []
    closed_res.append({"model": "SupCon", **calc_closed(closed_df["supcon_predicted_taxon"], closed_df["true_taxon"])})
    closed_res.append({"model": "Baseline", **calc_closed(closed_df["baseline_predicted_taxon"], closed_df["true_taxon"])})
    pd.DataFrame(closed_res).to_csv("closed_set_results.csv", index=False)
    
    # 6. Open-Set Metrics (Binary classification: Novel vs Known)
    def calc_open(pred_novel, nov_score, y_true):
        fpr, tpr, thresholds = roc_curve_proxy(y_true, nov_score)
        
        return {
            "AUROC": roc_auc_score(y_true, nov_score),
            "AUPRC": average_precision_score(y_true, nov_score),
            "FPR@95TPR": get_fpr_at_95_tpr(fpr, tpr),
            "novelty_precision": precision_score(y_true, pred_novel, zero_division=0),
            "novelty_recall": recall_score(y_true, pred_novel, zero_division=0),
            "novelty_f1": f1_score(y_true, pred_novel, zero_division=0)
        }
        
    def roc_curve_proxy(y, score):
        from sklearn.metrics import roc_curve
        return roc_curve(y, score)
        
    def get_fpr_at_95_tpr(fpr, tpr):
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            return fpr[idx[0]]
        return 1.0

    open_res = []
    open_res.append({"model": "SupCon", **calc_open(df["supcon_predicted_novel"], df["supcon_novelty_score"], df["true_novel"])})
    open_res.append({"model": "Baseline", **calc_open(df["baseline_predicted_novel"], df["baseline_novelty_score"], df["true_novel"])})
    pd.DataFrame(open_res).to_csv("open_set_results.csv", index=False)
    
    # 7. Confusion Matrix
    # We create a binary Known vs Novel confusion matrix for the open-set evaluation
    cm_sup = confusion_matrix(df["true_novel"], df["supcon_predicted_novel"])
    cm_base = confusion_matrix(df["true_novel"], df["baseline_predicted_novel"])
    
    with open("confusion_matrix.csv", "w") as f:
        f.write("Model,TrueClass,PredictedClass,Count\n")
        labels = ["Known", "Novel"]
        for i in range(2):
            for j in range(2):
                f.write(f"SupCon,{labels[i]},{labels[j]},{cm_sup[i,j]}\n")
                f.write(f"Baseline,{labels[i]},{labels[j]},{cm_base[i,j]}\n")

    print("Phase 3 Evaluation Diagnosis Complete!")
    print("Generated: prediction_audit.csv, closed_set_results.csv, open_set_results.csv, confusion_matrix.csv")

if __name__ == "__main__":
    main()
