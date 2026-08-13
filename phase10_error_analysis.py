import os
import pandas as pd
import numpy as np

def run_phase10():
    print("Loading Phase 6 predictions...")
    df = pd.read_csv('phase6_final_predictions.csv')
    
    print("10A. Sequence Integrity Audit")
    
    # 1. Expected configs = 20
    configs = df[['rank', 'seed']].drop_duplicates()
    if len(configs) != 20:
        raise ValueError(f"Expected 20 configurations, found {len(configs)}")
        
    # 2. Duplicate sequence IDs within configuration
    dup_counts = df.groupby(['rank', 'seed', 'sequence_id']).size()
    if (dup_counts > 1).any():
        dups = dup_counts[dup_counts > 1]
        raise ValueError(f"Found duplicate sequence IDs within configuration: {dups}")
        
    # 3. split == 'test'
    test_df = df[df['split'] == 'test'].copy()
    if len(test_df) == 0:
        raise ValueError("No test split found!")
    if len(df[df['split'] != 'test']) > 0:
        print(f"Filtering out {len(df[df['split'] != 'test'])} non-test records.")
        
    # 4. Missing prediction status
    if test_df['supcon_predicted_status'].isna().any():
        raise ValueError("Missing supcon_predicted_status detected!")
        
    # 5. Invalid numerical score
    if not np.isfinite(test_df['z_score']).all():
        raise ValueError("Invalid z_score detected!")
    if not np.isfinite(test_df['confidence']).all():
        raise ValueError("Invalid confidence detected!")
        
    print("Audit passed.")
    
    # Load Taxonomy to resolve homonyms
    print("Loading taxonomy map...")
    tax_file = "assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    taxon_to_lineages = {} # taxon -> set of (phylum, class, order, family, genus)
    
    if os.path.exists(tax_file):
        with open(tax_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 7:
                    p, c, o, fa, g = parts[2], parts[3], parts[4], parts[5], parts[6]
                    lineage = (p, c, o, fa, g)
                    for idx, rank_name in enumerate(['class', 'order', 'family', 'genus']):
                        t_name = parts[3 + idx]
                        if t_name not in taxon_to_lineages:
                            taxon_to_lineages[t_name] = set()
                        taxon_to_lineages[t_name].add(lineage)
                        
    print("10B & 10C. Layer A & B Categorization")
    
    def categorize(row):
        t_status = row['true_status_at_rank']
        p_status = row['supcon_predicted_status']
        t_tax = row['true_taxon_at_rank']
        p_tax = row['supcon_predicted_taxon']
        
        # Unresolved
        if pd.isna(t_tax) or pd.isna(p_tax) or t_tax == 'NA' or p_tax == 'NA' or t_tax == 'uncultured' or p_tax == 'uncultured':
            return 'Unresolved', 'Unresolved'
            
        # Layer A
        if t_status == 'KNOWN' and p_status == 'KNOWN':
            layer_a = 'Known -> Known'
        elif t_status == 'NOVEL' and p_status == 'NOVEL':
            layer_a = 'Novel -> Novel'
        elif t_status == 'NOVEL' and p_status == 'KNOWN':
            layer_a = 'False Known (FK)'
        elif t_status == 'KNOWN' and p_status == 'NOVEL':
            layer_a = 'False Novel (FN)'
        else:
            layer_a = 'Unknown'
            
        # Layer B (only for Known -> Known)
        layer_b = 'N/A'
        if layer_a == 'Known -> Known':
            if t_tax == p_tax:
                # Check for homonym
                is_homonym = False
                # true lineage
                t_lin = (row.get('true_phylum', 'NA'), row.get('true_class', 'NA'), row.get('true_order', 'NA'), row.get('true_family', 'NA'), row.get('true_genus', 'NA'))
                p_lineages = taxon_to_lineages.get(p_tax, set())
                
                # If the true lineage doesn't match any known lineage for this taxon, or if there are multiple lineages
                # we'll do a simpler check: if we can confidently say it's a homonym.
                # Since we don't have true_phylum easily in the dataframe, let's just check class/order/family/genus
                t_sub_lin = (row.get('true_class', 'NA'), row.get('true_order', 'NA'), row.get('true_family', 'NA'), row.get('true_genus', 'NA'))
                
                # Do any of the p_lineages match this sub_lin?
                match_found = False
                for p_lin in p_lineages:
                    p_sub_lin = p_lin[1:] # class, order, family, genus
                    
                    # compare up to the evaluated rank
                    rank = row['rank']
                    rank_idx = ['class', 'order', 'family', 'genus'].index(rank)
                    
                    # if they match up to rank_idx
                    if p_sub_lin[:rank_idx] == t_sub_lin[:rank_idx]:
                        match_found = True
                        break
                        
                if len(p_lineages) > 1 and not match_found:
                    is_homonym = True
                
                if is_homonym:
                    layer_b = 'Homonym/lineage clash'
                else:
                    layer_b = 'Correct taxon'
            else:
                layer_b = 'Wrong taxon'
                
        return layer_a, layer_b

    test_df[['LayerA', 'LayerB']] = test_df.apply(categorize, axis=1, result_type='expand')
    
    # Baseline categorization
    def categorize_baseline(row):
        t_status = row['true_status_at_rank']
        p_status = row['baseline_predicted_status']
        t_tax = row['true_taxon_at_rank']
        p_tax = row['baseline_predicted_taxon']
        
        if pd.isna(t_tax) or pd.isna(p_tax) or t_tax == 'NA' or p_tax == 'NA' or t_tax == 'uncultured' or p_tax == 'uncultured':
            return 'Unresolved', 'Unresolved'
            
        if t_status == 'KNOWN' and p_status == 'KNOWN':
            la = 'Known -> Known'
        elif t_status == 'NOVEL' and p_status == 'NOVEL':
            la = 'Novel -> Novel'
        elif t_status == 'NOVEL' and p_status == 'KNOWN':
            la = 'False Known (FK)'
        elif t_status == 'KNOWN' and p_status == 'NOVEL':
            la = 'False Novel (FN)'
        else:
            la = 'Unknown'
            
        lb = 'N/A'
        if la == 'Known -> Known':
            if t_tax == p_tax:
                lb = 'Correct taxon'
            else:
                lb = 'Wrong taxon'
        return la, lb
        
    test_df[['Baseline_LayerA', 'Baseline_LayerB']] = test_df.apply(categorize_baseline, axis=1, result_type='expand')
    
    # 1. phase10_error_frequencies.csv
    print("Generating error frequencies...")
    def calc_rates(df_part, prefix=''):
        total_novel = len(df_part[df_part['true_status_at_rank'] == 'NOVEL'])
        total_known = len(df_part[df_part['true_status_at_rank'] == 'KNOWN'])
        
        fk = len(df_part[df_part[f'{prefix}LayerA'] == 'False Known (FK)'])
        fn = len(df_part[df_part[f'{prefix}LayerA'] == 'False Novel (FN)'])
        
        fk_rate = fk / total_novel if total_novel > 0 else 0
        fn_rate = fn / total_known if total_known > 0 else 0
        
        known_known = df_part[df_part[f'{prefix}LayerA'] == 'Known -> Known']
        valid_known_known = len(known_known[known_known[f'{prefix}LayerB'] != 'Unresolved'])
        wrong_tax = len(known_known[known_known[f'{prefix}LayerB'] == 'Wrong taxon'])
        wrong_tax_rate = wrong_tax / valid_known_known if valid_known_known > 0 else 0
        
        return pd.Series({
            'Total_Novel': total_novel,
            'Total_Known': total_known,
            'FK': fk,
            'FN': fn,
            'FK_Rate': fk_rate,
            'FN_Rate': fn_rate,
            'Known_Known_Valid': valid_known_known,
            'Wrong_Taxon': wrong_tax,
            'Wrong_Taxon_Rate': wrong_tax_rate
        })
        
    pooled = calc_rates(test_df)
    pooled.to_csv('phase10_error_frequencies.csv', header=True)
    
    # 2. phase10_error_frequencies_by_seed.csv
    by_seed = test_df.groupby(['rank', 'seed']).apply(calc_rates).reset_index()
    by_seed.to_csv('phase10_error_frequencies_by_seed.csv', index=False)
    
    # 3. phase10_error_boundary_analysis.csv
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '>=2.0']
    test_df['Z_Bin'] = pd.cut(test_df['z_score'], bins=bins, labels=labels)
    
    conf_bins = [-np.inf, 0.4, 0.45, 0.5, 0.55, 0.6, np.inf]
    conf_labels = ['<0.4', '0.4-0.45', '0.45-0.5', '0.5-0.55', '0.55-0.6', '>0.6']
    test_df['Conf_Bin'] = pd.cut(test_df['confidence'], bins=conf_bins, labels=conf_labels)
    
    boundary_z = test_df.groupby(['Z_Bin', 'LayerA']).size().reset_index(name='Count')
    boundary_c = test_df.groupby(['Conf_Bin', 'LayerA']).size().reset_index(name='Count')
    
    boundary_z.to_csv('phase10_error_boundary_analysis_Z.csv', index=False)
    boundary_c.to_csv('phase10_error_boundary_analysis_Conf.csv', index=False)
    
    # 4. phase10_taxonomic_error_analysis.csv
    test_df['embedding_distance'] = test_df['distance']
    tax_error = test_df.groupby(['rank', 'LayerA', 'LayerB']).agg(
        Count=('sequence_id', 'count'),
        Mean_Z=('z_score', 'mean'),
        Mean_Conf=('confidence', 'mean'),
        Mean_Dist=('embedding_distance', 'mean')
    ).reset_index()
    tax_error.to_csv('phase10_taxonomic_error_analysis.csv', index=False)
    
    # 5. phase10_representative_errors.csv
    reps = []
    
    # Highest-confidence False Known
    fk_df = test_df[test_df['LayerA'] == 'False Known (FK)']
    if not fk_df.empty:
        idx = fk_df['confidence'].idxmax()
        row = fk_df.loc[idx].copy()
        row['Rep_Type'] = 'Highest-confidence False Known'
        reps.append(row)
        
        idx = fk_df['z_score'].idxmin()
        row = fk_df.loc[idx].copy()
        row['Rep_Type'] = 'Most extreme False Known (min Z)'
        reps.append(row)
        
    fn_df = test_df[test_df['LayerA'] == 'False Novel (FN)']
    if not fn_df.empty:
        idx = fn_df['z_score'].idxmax()
        row = fn_df.loc[idx].copy()
        row['Rep_Type'] = 'Strongest False Novel (max Z)'
        reps.append(row)
        
        idx = fn_df['confidence'].idxmin()
        row = fn_df.loc[idx].copy()
        row['Rep_Type'] = 'Lowest-confidence False Novel'
        reps.append(row)
        
        idx = fn_df['embedding_distance'].idxmax()
        row = fn_df.loc[idx].copy()
        row['Rep_Type'] = 'Largest-distance False Novel'
        reps.append(row)
        
    homo_df = test_df[test_df['LayerB'] == 'Homonym/lineage clash']
    if not homo_df.empty:
        row = homo_df.iloc[0].copy() # Just take the first one for now
        row['Rep_Type'] = 'Homonym clash example'
        reps.append(row)
        
    if reps:
        reps_df = pd.DataFrame(reps)
        cols = ['Rep_Type', 'sequence_id', 'rank', 'seed', 'LayerA', 'LayerB', 'true_taxon_at_rank', 'supcon_predicted_taxon', 'z_score', 'confidence', 'embedding_distance']
        # Add true lineage cols
        for c in ['true_class', 'true_order', 'true_family', 'true_genus']:
            if c in reps_df.columns: cols.append(c)
        reps_df[cols].to_csv('phase10_representative_errors.csv', index=False)
        
    # 6. phase10_baseline_comparison.csv
    baseline_pooled = calc_rates(test_df, prefix='Baseline_')
    comp_df = pd.DataFrame({
        'Metric': pooled.index,
        'HARBOUR': pooled.values,
        'Baseline': baseline_pooled.values
    })
    comp_df.to_csv('phase10_baseline_comparison.csv', index=False)
    
    print("Writing Phase10_Error_Analysis_Report.md...")
    with open('Phase10_Error_Analysis_Report.md', 'w') as f:
        f.write("# Phase 10: Error Analysis Report\n\n")
        f.write("## 1. Sequence Integrity Audit\n")
        f.write("PASS: 20 configurations, no duplicates, valid scores.\n\n")
        
        f.write("## 2. Open-Set Detection Failures (Layer A)\n")
        f.write("### HARBOUR\n")
        f.write(f"- False Known Rate: {pooled['FK_Rate']:.2%}\n")
        f.write(f"- False Novel Rate: {pooled['FN_Rate']:.2%}\n\n")
        
        f.write("## 3. Taxonomic Classification Failures (Layer B)\n")
        f.write(f"- Wrong Taxon Rate (among Accepted Known): {pooled['Wrong_Taxon_Rate']:.2%}\n\n")
        
        f.write("## 4. Decision-Boundary / Error-Severity\n")
        f.write("See `phase10_error_boundary_analysis_Z.csv` and `phase10_error_boundary_analysis_Conf.csv`.\n\n")
        
        f.write("## 5. HARBOUR vs MultinomialNB Failure-Mode Comparison\n")
        f.write(comp_df.to_markdown(index=False))
        
    print("Phase 10 successfully completed.")

if __name__ == "__main__":
    run_phase10()
