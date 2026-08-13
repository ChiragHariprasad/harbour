import pandas as pd

df_seed = pd.read_csv('phase11_seed_statistics.csv')
with open('Phase11_Statistical_Report.md', 'w', encoding='utf-8') as f:
    f.write('# Phase 11: Statistical Analysis Report\n\n')
    f.write('## 11A Configuration-level bootstrap\n')
    f.write('See `phase11_configuration_statistics.csv` for detailed point estimates and sequence-level uncertainty CIs for all 20 configurations.\n\n')
    
    f.write('## 11B Experimental Reproducibility (Seed-level)\n')
    f.write('Seed-level variation across the 5 independent training seeds (Mean ± SD).\n\n')
    
    for rank in ['family', 'genus', 'order', 'class']:
        if rank in ['family', 'genus']:
            f.write(f'### {rank.capitalize()} (Open-Set Evaluation)\n')
        else:
            f.write(f'### {rank.capitalize()} (Hierarchical Generalization)\n')
            
        rank_rows = df_seed[df_seed['rank'] == rank]
        for m in ['H_Score', 'AUROC', 'A_known', 'R_novel']:
            h_metric = f'HARBOUR_{m}'
            row = rank_rows[rank_rows['metric'] == h_metric].iloc[0]
            f.write(f'- HARBOUR {m}: {row["mean_across_seeds"]:.4f} ± {row["sd_across_seeds"]:.4f}\n')
        f.write('\n')
        
    f.write('## 11C Paired Comparative Inference\n')
    f.write('Statistically distinguishable improvement at α=0.05 is achieved if the 95% CI of Δ excludes zero. Detailed results in `phase11_comparative_statistics.csv`.\n\n')
    
    f.write('## Statistical Caveat\n')
    f.write('> [!NOTE]\n')
    f.write('> Sequence-level bootstrap CIs quantify prediction-level uncertainty and do not constitute independent biological replicates. Seed-level variation provides the complementary measure of experimental reproducibility.\n')
print('Report Generated!')
