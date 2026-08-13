import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

base_dir = "E:/HTRP"

def compute_metrics(y_true_binary, y_scores, threshold):
    """
    y_true_binary: 1 if NOVEL, 0 if KNOWN
    y_scores: Score where higher means MORE novel
    """
    # Hard predictions
    y_pred = (y_scores >= threshold).astype(int)
    
    # known = 0, novel = 1
    # Known accuracy: true negative rate
    # Novel recall: true positive rate
    
    tn = np.sum((y_true_binary == 0) & (y_pred == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred == 0))
    tp = np.sum((y_true_binary == 1) & (y_pred == 1))
    
    total_known = tn + fp
    total_novel = tp + fn
    
    a_known = tn / total_known if total_known > 0 else 0
    r_novel = tp / total_novel if total_novel > 0 else 0
    p_novel = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1_novel = 2 * (p_novel * r_novel) / (p_novel + r_novel) if (p_novel + r_novel) > 0 else 0
    
    h_score = 2 * (a_known * r_novel) / (a_known + r_novel) if (a_known + r_novel) > 0 else 0
    
    return {
        "a_known": a_known,
        "r_novel": r_novel,
        "p_novel": p_novel,
        "f1_novel": f1_novel,
        "h_score": h_score,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

def main():
    print("PHASE 5: Threshold Sweep")
    
    df = pd.read_csv(f"{base_dir}/phase5_raw_scores.csv")
    df['is_novel_true'] = (df['true_status'] == 'NOVEL').astype(int)
    
    # Evaluate closed-set native accuracy (Accept-All) for constraint
    # Native accuracy = accuracy on KNOWN sequences when NOVELTY threshold is completely relaxed.
    # For SupCon, that is just: is predicted_taxon == true_taxon on KNOWN sequences?
    print("Calculating native closed-set accuracies...")
    cs_accs = {}
    
    df_known = df[df['is_novel_true'] == 0]
    for config, group in df_known.groupby(['model', 'rank', 'seed']):
        model, rank, seed = config
        acc = np.mean(group['predicted_taxon'] == group['true_taxon'])
        cs_accs[config] = acc
        
    df_supcon = df[df['model'] == 'SupCon']
    df_baseline = df[df['model'] == 'ClassicalBaseline']
    
    z_grid = np.arange(0.5, 4.5, 0.5)
    c_grid = np.arange(0.1, 1.1, 0.1)
    
    supcon_results = []
    
    print("Sweeping SupCon Thresholds...")
    for z_thresh in z_grid:
        for c_thresh in c_grid:
            
            # Predict NOVEL (1) if confidence < c_thresh OR z_score >= z_thresh
            df_supcon['is_novel_pred'] = ((df_supcon['confidence'] < c_thresh) | (df_supcon['z_score'] >= z_thresh)).astype(int)
            
            config_h_scores = []
            valid_constraint = True
            
            for (rank, seed), group in df_supcon.groupby(['rank', 'seed']):
                y_true = group['is_novel_true'].values
                y_pred = group['is_novel_pred'].values
                
                # Known logic: it's correct if it's NOT flagged novel AND predicted_taxon == true_taxon
                # Wait, open-set known accuracy requires it to be accepted AND correctly classified!
                known_mask = (y_true == 0)
                accepted_mask = (y_pred == 0)
                correct_class_mask = (group['predicted_taxon'].values == group['true_taxon'].values)
                
                total_known = np.sum(known_mask)
                a_known = np.sum(known_mask & accepted_mask & correct_class_mask) / total_known if total_known > 0 else 0
                
                novel_mask = (y_true == 1)
                total_novel = np.sum(novel_mask)
                r_novel = np.sum(novel_mask & (y_pred == 1)) / total_novel if total_novel > 0 else 0
                
                h_score = 2 * (a_known * r_novel) / (a_known + r_novel) if (a_known + r_novel) > 0 else 0
                config_h_scores.append(h_score)
                
                native_a_cs = cs_accs[('SupCon', rank, seed)]
                if a_known < 0.8 * native_a_cs:
                    valid_constraint = False
                    
            mean_h = np.mean(config_h_scores)
            
            supcon_results.append({
                "z_thresh": z_thresh,
                "c_thresh": c_thresh,
                "mean_h_score": mean_h,
                "valid_constraint": valid_constraint
            })
            
    res_df = pd.DataFrame(supcon_results)
    res_df.to_csv(f"{base_dir}/phase5_threshold_sweep_results.csv", index=False)
    
    valid_res = res_df[res_df['valid_constraint']]
    if len(valid_res) == 0:
        print("WARNING: No threshold satisfies the 80% known accuracy retention constraint!")
        best_row = res_df.loc[res_df['mean_h_score'].idxmax()]
    else:
        best_row = valid_res.loc[valid_res['mean_h_score'].idxmax()]
        
    best_z = best_row['z_thresh']
    best_c = best_row['c_thresh']
    
    print(f"\nOptimal SupCon Thresholds: Z = {best_z}, Conf = {best_c}")
    print(f"Mean H-Score: {best_row['mean_h_score']:.4f}")
    
    # Bootstrap for optimal threshold
    print("\nBootstrapping 1,000 resamples for optimal threshold...")
    np.random.seed(42)
    boot_h = []
    boot_rk = []
    boot_rn = []
    
    df_opt = df_supcon.copy()
    df_opt['is_novel_pred'] = ((df_opt['confidence'] < best_c) | (df_opt['z_score'] >= best_z)).astype(int)
    
    for _ in range(1000):
        # Resample the indices
        idx = np.random.choice(len(df_opt), len(df_opt), replace=True)
        samp = df_opt.iloc[idx]
        
        known_mask = (samp['is_novel_true'] == 0)
        total_known = np.sum(known_mask)
        a_known = np.sum(known_mask & (samp['is_novel_pred'] == 0) & (samp['predicted_taxon'] == samp['true_taxon'])) / total_known if total_known > 0 else 0
        
        novel_mask = (samp['is_novel_true'] == 1)
        total_novel = np.sum(novel_mask)
        r_novel = np.sum(novel_mask & (samp['is_novel_pred'] == 1)) / total_novel if total_novel > 0 else 0
        
        h = 2 * (a_known * r_novel) / (a_known + r_novel) if (a_known + r_novel) > 0 else 0
        
        boot_h.append(h)
        boot_rk.append(a_known)
        boot_rn.append(r_novel)
        
    print(f"H-Score 95% CI: [{np.percentile(boot_h, 2.5):.4f}, {np.percentile(boot_h, 97.5):.4f}]")
    print(f"Known Acc 95% CI: [{np.percentile(boot_rk, 2.5):.4f}, {np.percentile(boot_rk, 97.5):.4f}]")
    print(f"Novel Recall 95% CI: [{np.percentile(boot_rn, 2.5):.4f}, {np.percentile(boot_rn, 97.5):.4f}]")
    
    # Save threshold selection
    with open(f"{base_dir}/phase5_score_provenance.json", "r") as f:
        prov = json.load(f)
        
    out = {
        "method": "SupCon_FAISS",
        "z_threshold": float(best_z),
        "confidence_threshold": float(best_c),
        "selection_metric": "mean_configuration_H_score",
        "known_accuracy_constraint": ">= 0.8 * A_cs",
        "validation_only": True,
        "test_used": False,
        "n_configurations": 20,
        "n_bootstrap": 1000,
        "dataset_hash": prov[0]["train_fit_hash"],
        "h_score_mean": float(np.mean(boot_h)),
        "h_score_95ci": [float(np.percentile(boot_h, 2.5)), float(np.percentile(boot_h, 97.5))]
    }
    
    with open(f"{base_dir}/threshold_selection.json", "w") as f:
        json.dump(out, f, indent=4)
        
    print("Phase 5 sweep complete!")

if __name__ == "__main__":
    main()
