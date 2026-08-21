import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix

base_dir = "E:/paper_revamp"

ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

models = [
    'MultinomialNB',
    'MSP-full',
    'VSEARCH',
    'Kraken2',
    'DNABERT-S',
    'Nucleotide Transformer',
    'HARBOUR'
]

# Helper to compute metrics given arrays
def compute_metrics(y_true_cls, y_pred_cls, y_true_bin, y_pred_bin, y_score, true_status):
    # y_true_bin: 1 for Novel, 0 for Known
    # y_pred_bin: 1 for Novel, 0 for Known
    
    # H-score
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1]).ravel()
    a_known = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    r_novel = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    h_score = 2 * (a_known * r_novel) / (a_known + r_novel) if (a_known + r_novel) > 0 else 0.0
    
    # AUROC
    try:
        auroc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        auroc = np.nan
        
    # Multiclass labels
    # If truly known, it should predict the exact taxon. If truly novel, it should predict "NOVEL".
    # Wait, the user asked for: Closed-set F1, Novel F1, H-score, AUROC, MCC
    
    y_t_mc = np.where(y_true_bin == 1, "NOVEL", y_true_cls.astype(str))
    y_p_mc = np.where(y_pred_bin == 1, "NOVEL", y_pred_cls.astype(str))
    
    mcc = matthews_corrcoef(y_t_mc, y_p_mc)
    
    # Novel F1: Binary F1 for the Novel class
    p_novel = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    novel_f1 = 2 * (p_novel * r_novel) / (p_novel + r_novel) if (p_novel + r_novel) > 0 else 0.0
    
    # Closed-set F1: F1 on ONLY the known sequences (ignoring rejection!)
    # Wait, the user wants "Closed-set F1". Typically this means F1 computed ONLY on known instances.
    # Let's filter to only known instances.
    known_idx = (y_true_bin == 0)
    if np.sum(known_idx) > 0:
        y_t_cs = y_true_cls[known_idx].astype(str)
        y_p_cs = y_pred_cls[known_idx].astype(str)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs_f1 = f1_score(y_t_cs, y_p_cs, average='macro')
    else:
        cs_f1 = np.nan
        
    return {
        "H_score": h_score,
        "AUROC": auroc,
        "MCC": mcc,
        "Novel_F1": novel_f1,
        "Closed_set_F1": cs_f1,
        "A_known": a_known,
        "R_novel": r_novel
    }

def process_file(model, rank, seed):
    if model == 'MultinomialNB' or model == 'HARBOUR':
        df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
        df = df[(df['rank'] == rank) & (df['seed'] == seed)].copy()
        
        y_true_cls = df['true_taxon_at_rank'].values
        y_true_bin = (df['true_status_at_rank'] == 'NOVEL').astype(int).values
        
        if model == 'MultinomialNB':
            y_pred_cls = df['baseline_predicted_taxon'].values
            y_pred_bin = (df['baseline_predicted_status'] == 'NOVEL').astype(int).values
            y_score = 1.0 - df['baseline_probability'].values
        else:
            y_pred_cls = df['supcon_predicted_taxon'].values
            y_pred_bin = (df['supcon_predicted_status'] == 'NOVEL').astype(int).values
            y_score = df['z_score'].values
            
    elif model == 'MSP-full':
        path = f"{base_dir}/phase13_msp_baseline/{rank}/seed_{seed}/predictions.csv"
        df = pd.read_csv(path)
        y_true_cls = df['true_taxon'].values
        y_true_bin = (df['true_status'] == 'NOVEL').astype(int).values
        y_pred_cls = df['predicted_taxon'].values
        y_pred_bin = (df['predicted_status'] == 'NOVEL').astype(int).values
        y_score = 1.0 - df['msp_score'].values
        
    elif model == 'VSEARCH':
        path = f"{base_dir}/phase15_vsearch_baseline/{rank}/seed_{seed}/predictions.csv"
        df = pd.read_csv(path)
        y_true_cls = df['true_taxon'].values
        y_true_bin = (df['true_status'] == 'NOVEL').astype(int).values
        y_pred_cls = df['predicted_taxon'].values
        y_pred_bin = (df['predicted_status'] == 'NOVEL').astype(int).values
        y_score = -df['pident'].values
        
    elif model == 'Kraken2':
        path = f"{base_dir}/phase16_kraken2_baseline/db_{rank}_{seed}/predictions.csv"
        df = pd.read_csv(path)
        # Parse ground truth from the main test splits, wait, the kraken2 file has true_label
        # but what about true status? We need true status. Let's merge with ground truth or parse accession.
        # Actually, in kraken2 predictions: AF463459.1,Saturniidae,Saturniidae,0.3761,False
        # Wait, the true_label is the taxon. But is it novel?
        # We can read E:/HTRP/phase6_final_predictions.csv for this seed and merge by sequence_id/accession!
        gt_df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
        gt_df = gt_df[(gt_df['rank'] == rank) & (gt_df['seed'] == seed)][['sequence_id', 'true_taxon_at_rank', 'true_status_at_rank']]
        
        df = df.rename(columns={'accession': 'sequence_id'})
        df = pd.merge(df, gt_df, on='sequence_id', how='left')
        
        y_true_cls = df['true_taxon_at_rank'].values
        y_true_bin = (df['true_status_at_rank'] == 'NOVEL').astype(int).values
        y_pred_cls = df['pred_label'].values
        y_pred_bin = df['is_novel'].astype(int).values
        y_score = 1.0 - df['confidence'].values
        
    elif model in ['DNABERT-S', 'Nucleotide Transformer']:
        phase = "phase17_dnaberts_baseline" if model == 'DNABERT-S' else "phase18_nt_baseline"
        path = f"{base_dir}/{phase}/{rank}/seed_{seed}/predictions.csv"
        df = pd.read_csv(path)
        gt_df = pd.read_csv(f"E:/HTRP/phase6_final_predictions.csv")
        gt_df = gt_df[(gt_df['rank'] == rank) & (gt_df['seed'] == seed)][['sequence_id', 'true_taxon_at_rank', 'true_status_at_rank']]
        df = df.rename(columns={'accession': 'sequence_id'})
        df = pd.merge(df, gt_df, on='sequence_id', how='left')
        
        y_true_cls = df['true_taxon_at_rank'].values
        y_true_bin = (df['true_status_at_rank'] == 'NOVEL').astype(int).values
        y_pred_cls = df['pred_label'].values
        y_pred_bin = df['is_novel'].astype(int).values
        y_score = df['z_score'].values
        
    else:
        raise ValueError("Unknown model")
        
    return compute_metrics(y_true_cls, y_pred_cls, y_true_bin, y_pred_bin, y_score, df)

def main():
    print("Compiling Phase 21 Final Baseline Table...")
    all_results = []
    
    # We will import get_pooled_dataframe from phase21_bootstrap_baselines since it handles loading perfectly
    import sys
    import os
    from phase21_bootstrap_baselines import get_pooled_dataframe
    
    for rank in ranks:
        model_dfs = {m: get_pooled_dataframe(m, rank) for m in models}
        
        # PAIRED EVALUATION: exact intersection of (sequence_id, seed)
        for m in models:
            model_dfs[m]['pair_id'] = model_dfs[m]['sequence_id'].astype(str) + "_" + model_dfs[m]['seed'].astype(str)
            
        common_pairs = set(model_dfs[models[0]]['pair_id'])
        for m in models[1:]:
            common_pairs = common_pairs.intersection(set(model_dfs[m]['pair_id']))
            
        print(f"[{rank}] Computing exact means on {len(common_pairs)} paired instances")
        
        for model in models:
            df = model_dfs[model]
            df = df[df['pair_id'].isin(common_pairs)]
            df = df.sort_values('pair_id').reset_index(drop=True)
            
            # Since the metrics are computed PER SEED in compile_baselines (to get mean+-std across seeds)
            # We group by seed and compute
            seed_metrics = []
            for seed in seeds:
                sdf = df[df['seed'] == seed]
                mets = compute_metrics(
                    sdf['y_true_cls'].values,
                    sdf['y_pred_cls'].values,
                    sdf['y_true_bin'].values,
                    sdf['y_pred_bin'].values,
                    sdf['y_score'].values,
                    sdf
                )
                seed_metrics.append(mets)
                
            df_sm = pd.DataFrame(seed_metrics)
            row = {'Model': model, 'Rank': rank.capitalize()}
            for col in df_sm.columns:
                mean_val = df_sm[col].mean()
                std_val = df_sm[col].std()
                row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
                
            all_results.append(row)
            
    final_df = pd.DataFrame(all_results)
    out_csv = "E:/paper_revamp/phase21_final_baseline_table.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
