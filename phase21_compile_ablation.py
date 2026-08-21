import pandas as pd
import numpy as np

def main():
    print("Compiling Phase 21 Ablation Table...")
    
    df = pd.read_csv("E:/paper_revamp/phase14_ablation/phase14_ablation_summary.csv")
    
    # Ranks to include
    ranks = ['class', 'order', 'family', 'genus']
    configs = ['CNN_CE_MSP', 'CNN_SupCon_Dist', 'CNN_CE_DistConf', 'HARBOUR']
    
    # We want rows: Config x Rank
    # Columns: H-score, A_known, R_novel
    # Interpretation column based on config
    
    out_rows = []
    
    # Map configuration names to paper-friendly strings
    config_labels = {
        'CNN_CE_MSP': '(a) Cross-Entropy + MSP (Baseline)',
        'CNN_SupCon_Dist': '(b) SupCon + Dist-Only',
        'CNN_CE_DistConf': '(c) Cross-Entropy + Dist+Conf',
        'HARBOUR': '(d) HARBOUR (SupCon + Dist+Conf)'
    }
    
    # Map interpretation (Representation vs Decision)
    interpretation = {
        'CNN_CE_MSP': 'Baseline Mechanism',
        'CNN_SupCon_Dist': 'Representation Attribution (SupCon space isolates features, but distance alone may collapse)',
        'CNN_CE_DistConf': 'Decision Attribution (Distance/Confidence rejection applied to CE space)',
        'HARBOUR': 'Full System (Both)'
    }
    
    for config in configs:
        for rank in ranks:
            sub = df[(df['config'] == config) & (df['rank'] == rank)]
            if len(sub) == 0:
                continue
                
            row = sub.iloc[0]
            
            # Format Mean +/- Std
            h_str = f"{row['H_score_mean']:.4f} ± {row['H_score_std']:.4f}"
            ak_str = f"{row['A_known_mean']:.4f} ± {row['A_known_std']:.4f}"
            rn_str = f"{row['R_novel_mean']:.4f} ± {row['R_novel_std']:.4f}"
            
            out_rows.append({
                'Configuration': config_labels[config],
                'Rank': rank.capitalize(),
                'H-Score': h_str,
                'Accuracy (Known)': ak_str,
                'Recall (Novel)': rn_str,
                'Attribution Interpretation': interpretation[config]
            })
            
    out_df = pd.DataFrame(out_rows)
    out_csv = "E:/paper_revamp/phase21_ablation_table.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

if __name__ == "__main__":
    main()
