import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix

base_dir = "E:/HTRP"

def calc_novelty_metrics(y_true_binary, y_scores, y_pred):
    # known = 0, novel = 1
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()
    
    total_known = tn + fp
    total_novel = tp + fn
    
    a_known = tn / total_known if total_known > 0 else 0
    r_novel = tp / total_novel if total_novel > 0 else 0
    p_novel = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_novel = 2 * (p_novel * r_novel) / (p_novel + r_novel) if (p_novel + r_novel) > 0 else 0
    spec = a_known
    
    h_score = 2 * (a_known * r_novel) / (a_known + r_novel) if (a_known + r_novel) > 0 else 0
    
    try:
        auroc = roc_auc_score(y_true_binary, y_scores)
    except ValueError:
        auroc = np.nan
        
    try:
        auprc = average_precision_score(y_true_binary, y_scores)
    except ValueError:
        auprc = np.nan
        
    # FPR@95TPR
    # Find threshold where TPR is >= 0.95
    sorted_idx = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true_binary[sorted_idx]
    
    tpr_cumulative = np.cumsum(y_true_sorted) / total_novel if total_novel > 0 else np.zeros_like(y_true_sorted)
    idx_95 = np.where(tpr_cumulative >= 0.95)[0]
    if len(idx_95) > 0:
        threshold_idx = idx_95[0]
        fpr95 = np.sum(y_true_sorted[:threshold_idx+1] == 0) / total_known if total_known > 0 else np.nan
    else:
        fpr95 = np.nan
        
    return {
        "auroc": auroc,
        "auprc": auprc,
        "novel_recall": r_novel,
        "novel_precision": p_novel,
        "novel_f1": f1_novel,
        "fpr95": fpr95,
        "specificity": spec,
        "known_accuracy": a_known,
        "h_score": h_score
    }

def calc_multiclass_metrics(y_true_classes, y_pred_classes, y_true_binary, y_pred_binary):
    # For open-set multiclass, we need to map NOVEL to a special class e.g. "NOVEL"
    # The true classes already have 'NOVEL' for novel sequences if they were properly mapped, but wait!
    # The true_taxon_at_rank is the actual taxon for Known, and some taxon for Novel.
    # To compute Overall Accuracy, if a sequence is truly NOVEL, it must be predicted as NOVEL.
    # If it is truly KNOWN, it must be predicted as the correct KNOWN taxon.
    
    y_t = np.where(y_true_binary == 1, "NOVEL", y_true_classes)
    y_p = np.where(y_pred_binary == 1, "NOVEL", y_pred_classes)
    
    acc = accuracy_score(y_t, y_p)
    b_acc = balanced_accuracy_score(y_t, y_p)
    mcc = matthews_corrcoef(y_t, y_p)
    
    # We must suppress warnings for labels with no predicted samples
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macro_f1 = f1_score(y_t, y_p, average='macro')
        weighted_f1 = f1_score(y_t, y_p, average='weighted')
        
    return {
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": b_acc,
        "mcc": mcc
    }

def process_model(df, model_name):
    y_true_bin = (df['true_status_at_rank'] == 'NOVEL').astype(int).values
    y_pred_bin = (df[f'{model_name}_predicted_status'] == 'NOVEL').astype(int).values
    
    if model_name == 'supcon':
        # Score where higher means more novel: negative confidence or Z? 
        # Z-score is standardized distance. High Z = Novel. 
        # Let's use Z-score as the continuous anomaly score for AUROC.
        y_scores = df['z_score'].values
    else:
        # For baseline, max posterior probability. Low prob = Novel. 
        # So novelty score = 1.0 - max_p
        y_scores = 1.0 - df['baseline_probability'].values
        
    nov_mets = calc_novelty_metrics(y_true_bin, y_scores, y_pred_bin)
    
    mc_mets = calc_multiclass_metrics(
        df['true_taxon_at_rank'].astype(str).values,
        df[f'{model_name}_predicted_taxon'].astype(str).values,
        y_true_bin,
        y_pred_bin
    )
    
    res = {}
    res.update(nov_mets)
    res.update(mc_mets)
    return res

def main():
    print("Computing Phase 6 Metrics...")
    df = pd.read_csv(f"{base_dir}/phase6_final_predictions.csv")
    
    ranks = ['class', 'order', 'family', 'genus']
    
    # 1. Seed-level aggregation (Mean +/- SD)
    seed_results = []
    for (rank, seed), group in df.groupby(['rank', 'seed']):
        sup_res = process_model(group, 'supcon')
        base_res = process_model(group, 'baseline')
        
        row = {"rank": rank, "seed": seed}
        for k in sup_res.keys():
            row[f"supcon_{k}"] = sup_res[k]
            row[f"baseline_{k}"] = base_res[k]
            row[f"delta_{k}"] = sup_res[k] - base_res[k]
            
            # RI = (Harb - Base) / Base * 100
            # Be careful with division by zero
            base_val = base_res[k]
            if base_val and base_val > 0:
                row[f"ri_{k}"] = (sup_res[k] - base_val) / base_val * 100
            else:
                row[f"ri_{k}"] = 0.0
                
        seed_results.append(row)
        
    df_seed = pd.DataFrame(seed_results)
    
    rank_summaries = []
    for rank, group in df_seed.groupby('rank'):
        summary = {"rank": rank}
        for c in df_seed.columns:
            if c not in ['rank', 'seed']:
                summary[f"{c}_mean"] = group[c].mean()
                summary[f"{c}_std"] = group[c].std()
        rank_summaries.append(summary)
        
    df_rank = pd.DataFrame(rank_summaries)
    df_rank.to_csv(f"{base_dir}/phase6_metrics_by_rank.csv", index=False)
    
    # 2. Bootstrapping across pooled sequences per rank for CIs
    print("Bootstrapping 1000 resamples for sequence-level CIs...")
    np.random.seed(42)
    boot_results = []
    
    for rank, group in df.groupby('rank'):
        print(f"  Bootstrapping rank {rank}...")
        
        boot_metrics = {k: [] for k in ['h_score', 'novel_recall', 'known_accuracy', 'auroc', 'overall_accuracy']}
        
        for _ in range(1000):
            idx = np.random.choice(len(group), len(group), replace=True)
            samp = group.iloc[idx]
            
            sup_res = process_model(samp, 'supcon')
            
            for k in boot_metrics.keys():
                boot_metrics[k].append(sup_res[k])
                
        boot_row = {"rank": rank}
        for k, vals in boot_metrics.items():
            boot_row[f"{k}_mean"] = np.mean(vals)
            boot_row[f"{k}_2.5"] = np.percentile(vals, 2.5)
            boot_row[f"{k}_97.5"] = np.percentile(vals, 97.5)
            
        boot_results.append(boot_row)
        
    df_boot = pd.DataFrame(boot_results)
    df_boot.to_csv(f"{base_dir}/phase6_bootstrap_cis.csv", index=False)
    
    print("Done! Metrics saved.")

if __name__ == "__main__":
    main()
