import os
import json
import pandas as pd
import numpy as np

base_dir = "E:/HTRP"
p5_dir = f"{base_dir}/master Q1 details/Phase_5_Thresholds"

def main():
    print("Phase 6A: Sequential Freeze Protocol")
    
    # 1. Load Phase 5 scores
    print("Loading Phase 5 scores...")
    df = pd.read_csv(f"{p5_dir}/phase5_raw_scores.csv")
    
    df['is_novel_true'] = (df['true_status'] == 'NOVEL').astype(int)
    
    df_base = df[df['model'] == 'ClassicalBaseline'].copy()
    
    print("Calculating native closed-set accuracies for Baseline...")
    cs_accs = {}
    df_known = df_base[df_base['is_novel_true'] == 0]
    for config, group in df_known.groupby(['rank', 'seed']):
        rank, seed = config
        acc = np.mean(group['predicted_taxon'] == group['true_taxon'])
        cs_accs[config] = acc
        
    t_grid = np.arange(0.01, 1.0, 0.01)
    
    results = []
    print("Sweeping Baseline posterior thresholds...")
    for tau in t_grid:
        # Novel if max posterior probability < tau
        df_base['is_novel_pred'] = (df_base['confidence'] < tau).astype(int)
        
        config_h_scores = []
        valid_constraint = True
        
        for config, group in df_base.groupby(['rank', 'seed']):
            y_true = group['is_novel_true'].values
            y_pred = group['is_novel_pred'].values
            
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
            
            native_a_cs = cs_accs[config]
            if a_known < 0.8 * native_a_cs:
                valid_constraint = False
                
        mean_h = np.mean(config_h_scores)
        
        results.append({
            "tau": tau,
            "mean_h_score": mean_h,
            "valid_constraint": valid_constraint
        })
        
    res_df = pd.DataFrame(results)
    valid_res = res_df[res_df['valid_constraint']]
    if len(valid_res) == 0:
        print("WARNING: No baseline threshold satisfies the 80% constraint!")
        best_row = res_df.loc[res_df['mean_h_score'].idxmax()]
    else:
        best_row = valid_res.loc[valid_res['mean_h_score'].idxmax()]
        
    best_tau = best_row['tau']
    print(f"\nOptimal Baseline Threshold (tau): {best_tau:.2f}")
    print(f"Mean H-Score: {best_row['mean_h_score']:.4f}")
    
    # 3. Output baseline_threshold_selection.json
    base_json = {
        "method": "MultinomialNB",
        "statistic": "max_posterior_probability",
        "threshold": float(best_tau),
        "selection_dataset": "phase5_validation",
        "selection_metric": "mean_configuration_H_score",
        "test_used": False
    }
    with open(f"{base_dir}/baseline_threshold_selection.json", "w") as f:
        json.dump(base_json, f, indent=4)
        
    # 4. Output phase6_freeze_manifest.json
    # Read SupCon threshold
    with open(f"{p5_dir}/threshold_selection.json", "r") as f:
        supcon_prov = json.load(f)
        
    manifest = {
        "supcon_z_threshold": supcon_prov["z_threshold"],
        "supcon_confidence_threshold": supcon_prov["confidence_threshold"],
        "baseline_threshold": float(best_tau),
        "checkpoint_id": "HARBOUR_Embedding_v1",
        "dataset_hash": supcon_prov["dataset_hash"],
        "test_used_for_selection": False
    }
    
    with open(f"{base_dir}/phase6_freeze_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    print("Phase 6A Freeze Protocol Complete!")
    
if __name__ == "__main__":
    main()
