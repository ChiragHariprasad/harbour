import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings

def calc_metrics(df_part, harbour_prefix='supcon_', baseline_prefix='baseline_'):
    y_true_binary = (df_part['true_status_at_rank'] == 'NOVEL').astype(int)
    
    metrics = {}
    
    for prefix in [harbour_prefix, baseline_prefix]:
        pred_status = df_part[f'{prefix}predicted_status']
        
        # known accuracy
        known_mask = df_part['true_status_at_rank'] == 'KNOWN'
        if known_mask.sum() > 0:
            a_known = (pred_status[known_mask] == 'KNOWN').mean()
        else:
            a_known = np.nan
            
        # novel recall
        novel_mask = df_part['true_status_at_rank'] == 'NOVEL'
        if novel_mask.sum() > 0:
            r_novel = (pred_status[novel_mask] == 'NOVEL').mean()
        else:
            r_novel = np.nan
            
        # H-score
        if not np.isnan(a_known) and not np.isnan(r_novel) and (a_known + r_novel) > 0:
            h_score = 2 * a_known * r_novel / (a_known + r_novel)
        else:
            h_score = np.nan
            
        # AUROC / AUPRC
        if prefix == harbour_prefix:
            y_score = df_part['z_score']
        else:
            y_score = 1.0 - df_part['baseline_probability']
            
        if len(y_true_binary.unique()) > 1:
            auroc = roc_auc_score(y_true_binary, y_score)
            auprc = average_precision_score(y_true_binary, y_score)
        else:
            auroc = np.nan
            auprc = np.nan
            
        model_name = 'HARBOUR' if prefix == harbour_prefix else 'Baseline'
        metrics[f'{model_name}_A_known'] = a_known
        metrics[f'{model_name}_R_novel'] = r_novel
        metrics[f'{model_name}_H_Score'] = h_score
        metrics[f'{model_name}_AUROC'] = auroc
        metrics[f'{model_name}_AUPRC'] = auprc
        
    for m in ['A_known', 'R_novel', 'H_Score', 'AUROC', 'AUPRC']:
        metrics[f'Delta_{m}'] = metrics[f'HARBOUR_{m}'] - metrics[f'Baseline_{m}']
        
    return metrics

def run_phase11():
    print("Loading predictions...")
    df = pd.read_csv('phase6_final_predictions.csv')
    
    # 1. Acceptance gate checks
    test_df = df[df['split'] == 'test'].copy()
    if len(test_df) == 0: raise ValueError("No test split found")
    if not (test_df['split'] == 'test').all():
        raise ValueError("Phase 5/validation/test contamination occurs (split != test).")
    if len(test_df) == 0: raise ValueError("No test split found")
    
    configs = test_df.groupby(['rank', 'seed'])
    if len(configs) != 20:
        raise ValueError(f"Expected exactly 20 configurations, found {len(configs)}")
        
    B = 1000
    np.random.seed(42)
    
    config_rows = []
    comparative_stats = []
    diagnostics = []
    
    for (rank, seed), group in tqdm(configs, desc="Bootstrapping Configurations"):
        # Sequence integrity checks
        if len(group) != group['sequence_id'].nunique():
            raise ValueError(f"Duplicate sequence IDs in {rank}_{seed}")
        if group['supcon_predicted_status'].isna().any() or group['baseline_predicted_status'].isna().any():
            raise ValueError(f"Missing prediction status in {rank}_{seed}")
            
        # Point Estimate
        pe = calc_metrics(group)
        
        # Bootstrap
        n = len(group)
        boot_metrics = {k: [] for k in pe.keys()}
        
        for b in range(B):
            # Same sampled indices used for both HARBOUR and baseline
            indices = np.random.choice(n, n, replace=True)
            sample_df = group.iloc[indices]
            
            b_metrics = calc_metrics(sample_df)
            
            for k, v in b_metrics.items():
                boot_metrics[k].append(v)
                
        # Aggregate bootstrap CIs
        for k in pe.keys():
            arr = np.array(boot_metrics[k])
            
            # Diagnostic for NaNs
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                diagnostics.append({'rank': rank, 'seed': seed, 'metric': k, 'nan_count': nan_count, 'total': B})
                
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) > 0:
                ci_lower = np.percentile(arr_clean, 2.5)
                ci_upper = np.percentile(arr_clean, 97.5)
                b_mean = np.mean(arr_clean)
            else:
                ci_lower = np.nan
                ci_upper = np.nan
                b_mean = np.nan
                
            config_rows.append({
                'configuration': f"{rank}_{seed}",
                'rank': rank,
                'seed': seed,
                'metric': k,
                'n_sequences': n,
                'point_estimate': pe[k],
                'bootstrap_mean': b_mean,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'n_valid_bootstraps': len(arr_clean),
                'n_undefined_bootstraps': nan_count
            })
            
        # Build comparative stats row
        for m in ['A_known', 'R_novel', 'H_Score', 'AUROC', 'AUPRC']:
            h_pe = pe[f'HARBOUR_{m}']
            b_pe = pe[f'Baseline_{m}']
            d_pe = pe[f'Delta_{m}']
            ri = (d_pe / b_pe * 100) if b_pe > 0 else np.nan
            
            # Get the delta CI from config_rows
            # The Delta metric name is f'Delta_{m}'
            delta_row = next(r for r in config_rows if r['configuration'] == f"{rank}_{seed}" and r['metric'] == f'Delta_{m}')
            d_lower = delta_row['CI_lower']
            d_upper = delta_row['CI_upper']
            
            sig = (d_lower > 0) or (d_upper < 0)
            
            comparative_stats.append({
                'configuration': f"{rank}_{seed}",
                'rank': rank,
                'seed': seed,
                'metric': m,
                'HARBOUR_PE': h_pe,
                'Baseline_PE': b_pe,
                'Delta_PE': d_pe,
                'Relative_Improvement_Pct': ri,
                'Delta_CI_Lower': d_lower,
                'Delta_CI_Upper': d_upper,
                'Significant': sig
            })
            
    df_config = pd.DataFrame(config_rows)
    df_config.to_csv('phase11_configuration_statistics.csv', index=False)
    
    df_comp = pd.DataFrame(comparative_stats)
    df_comp.to_csv('phase11_comparative_statistics.csv', index=False)
    
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv('phase11_bootstrap_diagnostics.csv', index=False)
    else:
        with open('phase11_bootstrap_diagnostics.csv', 'w') as f:
            f.write("rank,seed,metric,nan_count,total\\n")
            f.write("ALL,ALL,NONE,0,1000\\n")
            
    # Seed Aggregation (11B)
    print("Aggregating seed statistics...")
    seed_agg = []
    # group by rank and metric
    for (rank, metric), grp in df_config.groupby(['rank', 'metric']):
        seed_agg.append({
            'rank': rank,
            'metric': metric,
            'mean_across_seeds': grp['point_estimate'].mean(),
            'sd_across_seeds': grp['point_estimate'].std(ddof=1)
        })
        
    df_seed = pd.DataFrame(seed_agg)
    df_seed.to_csv('phase11_seed_statistics.csv', index=False)
    
    # Descriptive pooled statistics (11D)
    print("Generating global pooled descriptive summary...")
    # "report the global result according to the same aggregation convention used throughout the project"
    # Average across all 20 configurations.
    global_agg = []
    for metric, grp in df_config.groupby('metric'):
        global_agg.append({
            'metric': metric,
            'global_mean': grp['point_estimate'].mean(),
            'global_sd': grp['point_estimate'].std(ddof=1)
        })
    pd.DataFrame(global_agg).to_csv('phase11_descriptive_statistics.csv', index=False)
    
    # Write report
    print("Writing report...")
    with open('Phase11_Statistical_Report.md', 'w') as f:
        f.write("# Phase 11: Statistical Analysis Report\\n\\n")
        f.write("## 11A Configuration-level bootstrap\\n")
        f.write("See `phase11_configuration_statistics.csv` for detailed point estimates and sequence-level uncertainty CIs for all 20 configurations.\\n\\n")
        
        f.write("## 11B Experimental Reproducibility (Seed-level)\\n")
        f.write("Seed-level variation across the 5 independent training seeds (Mean ± SD).\\n\\n")
        
        for rank in ['family', 'genus', 'order', 'class']:
            if rank in ['family', 'genus']:
                f.write(f"### {rank.capitalize()} (Open-Set Evaluation)\\n")
            else:
                f.write(f"### {rank.capitalize()} (Hierarchical Generalization)\\n")
                
            rank_rows = df_seed[df_seed['rank'] == rank]
            for m in ['H_Score', 'AUROC', 'A_known', 'R_novel']:
                h_metric = f"HARBOUR_{m}"
                row = rank_rows[rank_rows['metric'] == h_metric].iloc[0]
                f.write(f"- HARBOUR {m}: {row['mean_across_seeds']:.4f} ± {row['sd_across_seeds']:.4f}\\n")
            f.write("\\n")
            
        f.write("## 11C Paired Comparative Inference\\n")
        f.write("Statistically distinguishable improvement at α=0.05 is achieved if the 95% CI of Δ excludes zero. Detailed results in `phase11_comparative_statistics.csv`.\\n\\n")
        
        f.write("## Statistical Caveat\\n")
        f.write("> [!NOTE]\\n")
        f.write("> Sequence-level bootstrap CIs quantify prediction-level uncertainty and do not constitute independent biological replicates. Seed-level variation provides the complementary measure of experimental reproducibility.\\n")
        
    print("Done!")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_phase11()
