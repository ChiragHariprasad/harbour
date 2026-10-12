import pandas as pd

# 1. Main Table
df_ci = pd.read_csv('E:/HTRP/paper_fixes_for_TBCC/phase21_bootstrap_ci.csv')
df_base = pd.read_csv('E:/HTRP/paper_fixes_for_TBCC/phase21_baseline_table.csv')

ranks = ['Class', 'Order', 'Family', 'Genus']
models = ['MultinomialNB', 'MSP-full', 'VSEARCH', 'Kraken2', 'DNABERT-S', 'Nucleotide Transformer', 'HARBOUR']
metrics = ['Closed_set_F1', 'Novel_F1', 'H_score', 'AUROC', 'MCC']

with open('E:/HTRP/paper_fixes_for_TBCC/manuscript_tables.md', 'w') as f:
    f.write('## 1. Main Baseline Results (Paired-Bootstrap 95% CI vs HARBOUR)\n\n')
    f.write('Note: Positive values indicate HARBOUR outperformed the baseline by that margin.\n\n')
    for rank in ranks:
        f.write(f'### Rank: {rank}\n')
        f.write('| Model | Closed-set F1 | Novel F1 | H-score | AUROC | MCC |\n')
        f.write('|-------|---------------|----------|---------|-------|-----|\n')
        for model in models:
            row_vals = []
            for metric in metrics:
                if model == 'HARBOUR':
                    val = df_base[(df_base['Rank'] == rank) & (df_base['Model'] == model)][metric].values[0]
                    row_vals.append(val.replace(' ± ', ' +/- '))
                else:
                    val = df_ci[(df_ci['Rank'] == rank) & (df_ci['Model'] == model)][metric].values[0]
                    row_vals.append(val)
            f.write(f"| {model} | {' | '.join(row_vals)} |\n")
        f.write('\n')

# 2. Ablation Table
df_abl = pd.read_csv('E:/HTRP/paper_revamp/phase14_ablation/phase14_ablation_summary.csv')
with open('E:/HTRP/paper_fixes_for_TBCC/manuscript_tables.md', 'a') as f:
    f.write('## 2. Ablation Results (Mean +/- S.D.)\n\n')
    for rank in ranks:
        f.write(f'### Rank: {rank}\n')
        f.write('| Configuration | H-score | A_known | R_novel | MCC |\n')
        f.write('|---------------|---------|---------|---------|-----|\n')
        sub = df_abl[df_abl['rank'] == rank.lower()]
        for _, r in sub.iterrows():
            config = r['configuration']
            h = f"{r['H_Score_mean']:.4f} +/- {r['H_Score_std']:.4f}"
            ak = f"{r['A_known_mean']:.4f} +/- {r['A_known_std']:.4f}"
            rn = f"{r['R_novel_mean']:.4f} +/- {r['R_novel_std']:.4f}"
            mcc = f"{r['MCC_mean']:.4f} +/- {r['MCC_std']:.4f}"
            f.write(f"| {config} | {h} | {ak} | {rn} | {mcc} |\n")
        f.write('\n')
