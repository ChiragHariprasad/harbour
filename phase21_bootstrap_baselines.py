import os
import pandas as pd
import numpy as np
from phase21_compile_baselines import compute_metrics, process_file, models, ranks, seeds

base_dir = "E:/paper_revamp"

def get_pooled_dataframe(model, rank):
    # This reads all 5 seeds for a model/rank and concatenates them
    dfs = []
    for seed in seeds:
        if model == 'MultinomialNB' or model == 'HARBOUR':
            df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
            df = df[(df['rank'] == rank) & (df['seed'] == seed)].copy()
            if model == 'MultinomialNB':
                df['seed'] = seed
                df['y_true_cls'] = df['true_taxon_at_rank']
                df['y_true_bin'] = (df['true_status_at_rank'] == 'NOVEL').astype(int)
                df['y_pred_cls'] = df['baseline_predicted_taxon']
                df['y_pred_bin'] = (df['baseline_predicted_status'] == 'NOVEL').astype(int)
                df['y_score'] = 1.0 - df['baseline_probability']
            else:
                df['seed'] = seed
                df['y_true_cls'] = df['true_taxon_at_rank']
                df['y_true_bin'] = (df['true_status_at_rank'] == 'NOVEL').astype(int)
                df['y_pred_cls'] = df['supcon_predicted_taxon']
                df['y_pred_bin'] = (df['supcon_predicted_status'] == 'NOVEL').astype(int)
                df['y_score'] = df['z_score']
        elif model == 'MSP-full':
            df = pd.read_csv(f"{base_dir}/phase13_msp_baseline/{rank}/seed_{seed}/predictions.csv")
            df['seed'] = seed
            df['y_true_cls'] = df['true_taxon']
            df['y_true_bin'] = (df['true_status'] == 'NOVEL').astype(int)
            df['y_pred_cls'] = df['predicted_taxon']
            df['y_pred_bin'] = (df['predicted_status'] == 'NOVEL').astype(int)
            df['y_score'] = 1.0 - df['msp_score']
        elif model == 'VSEARCH':
            df = pd.read_csv(f"{base_dir}/phase15_vsearch_baseline/{rank}/seed_{seed}/predictions.csv")
            df['seed'] = seed
            df['y_true_cls'] = df['true_taxon']
            df['y_true_bin'] = (df['true_status'] == 'NOVEL').astype(int)
            df['y_pred_cls'] = df['predicted_taxon']
            df['y_pred_bin'] = (df['predicted_status'] == 'NOVEL').astype(int)
            df['y_score'] = -df['pident']
        elif model == 'Kraken2':
            df = pd.read_csv(f"{base_dir}/phase16_kraken2_baseline/db_{rank}_{seed}/predictions.csv")
            gt_df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
            gt_df = gt_df[(gt_df['rank'] == rank) & (gt_df['seed'] == seed)][['sequence_id', 'true_taxon_at_rank', 'true_status_at_rank']]
            df = df.rename(columns={'accession': 'sequence_id'})
            df = pd.merge(df, gt_df, on='sequence_id', how='left')
            df['seed'] = seed
            df['y_true_cls'] = df['true_taxon_at_rank']
            df['y_true_bin'] = (df['true_status_at_rank'] == 'NOVEL').astype(int)
            df['y_pred_cls'] = df['pred_label']
            df['y_pred_bin'] = df['is_novel'].astype(int)
            df['y_score'] = 1.0 - df['confidence']
        elif model in ['DNABERT-S', 'Nucleotide Transformer']:
            phase = "phase17_dnaberts_baseline" if model == 'DNABERT-S' else "phase18_nt_baseline"
            df = pd.read_csv(f"{base_dir}/{phase}/{rank}/seed_{seed}/predictions.csv")
            gt_df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
            gt_df = gt_df[(gt_df['rank'] == rank) & (gt_df['seed'] == seed)][['sequence_id', 'true_taxon_at_rank', 'true_status_at_rank']]
            df = df.rename(columns={'accession': 'sequence_id'})
            df = pd.merge(df, gt_df, on='sequence_id', how='left')
            df['seed'] = seed
            df['y_true_cls'] = df['true_taxon_at_rank']
            df['y_true_bin'] = (df['true_status_at_rank'] == 'NOVEL').astype(int)
            df['y_pred_cls'] = df['pred_label']
            df['y_pred_bin'] = df['is_novel'].astype(int)
            df['y_score'] = df['z_score']
            
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def main():
    print("Running Sequence-level Bootstrap for ALL Baselines...")
    np.random.seed(42)
    boot_results = []
    
    for rank in ranks:
        print(f"Bootstrapping Rank: {rank}")
        model_dfs = {m: get_pooled_dataframe(m, rank) for m in models}
        
        # PAIRED EVALUATION: Find exact intersection of (sequence_id, seed) across all models
        for m in models:
            model_dfs[m]['pair_id'] = model_dfs[m]['sequence_id'].astype(str) + "_" + model_dfs[m]['seed'].astype(str)
            
        common_pairs = set(model_dfs[models[0]]['pair_id'])
        for m in models[1:]:
            common_pairs = common_pairs.intersection(set(model_dfs[m]['pair_id']))
            
        print(f"  -> {len(common_pairs)} common paired instances for {rank}")
        
        # Filter all dfs to exactly the common sequences and sort them so indices match perfectly
        for m in models:
            df = model_dfs[m]
            df = df[df['pair_id'].isin(common_pairs)]
            df = df.sort_values('pair_id').reset_index(drop=True)
            model_dfs[m] = df
            
        n_len = len(model_dfs['HARBOUR'])
        
        boot_metrics = {m: {k: [] for k in ['H_score', 'Closed_set_F1', 'Novel_F1', 'AUROC', 'MCC']} for m in models}
        
        # Pre-extract numpy arrays for blazing fast indexing
        model_arrays = {}
        for model in models:
            df = model_dfs[model]
            model_arrays[model] = {
                'y_true_cls': df['y_true_cls'].values,
                'y_pred_cls': df['y_pred_cls'].values,
                'y_true_bin': df['y_true_bin'].values,
                'y_pred_bin': df['y_pred_bin'].values,
                'y_score': df['y_score'].values
            }
            
        for b in range(1000):
            idx = np.random.choice(n_len, n_len, replace=True)
            for model in models:
                arrs = model_arrays[model]
                # We can pass None for 'df' since compute_metrics doesn't use it except maybe for custom stuff?
                # Wait, compute_metrics takes 'df' but doesn't actually use it for the core metrics!
                # Let's check compute_metrics in phase21_compile_baselines.py. It does not use 'df' if not needed, 
                # but wait, compute_metrics requires 6 arguments. We can just pass None.
                mets = compute_metrics(
                    arrs['y_true_cls'][idx],
                    arrs['y_pred_cls'][idx],
                    arrs['y_true_bin'][idx],
                    arrs['y_pred_bin'][idx],
                    arrs['y_score'][idx],
                    None
                )
                for k in boot_metrics[model].keys():
                    boot_metrics[model][k].append(mets[k])
                    
        for model in models:
            row = {'Model': model, 'Rank': rank.capitalize()}
            for k, vals in boot_metrics[model].items():
                mean_val = np.mean(vals)
                lb = np.percentile(vals, 2.5)
                ub = np.percentile(vals, 97.5)
                # Manuscript formatting: Mean [95% CI]
                row[k] = f"{mean_val:.3f} [{lb:.3f}-{ub:.3f}]"
            boot_results.append(row)
            
    df_boot = pd.DataFrame(boot_results)
    out_csv = "E:/paper_revamp/phase21_final_bootstrap_table.csv"
    df_boot.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
