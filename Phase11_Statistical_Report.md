# Phase 11: Statistical Analysis Report

## 11A Configuration-level bootstrap
See `phase11_configuration_statistics.csv` for detailed point estimates and sequence-level uncertainty CIs for all 20 configurations.

## 11B Experimental Reproducibility (Seed-level)
Seed-level variation across the 5 independent training seeds (Mean ± SD).

### Family (Open-Set Evaluation)
- HARBOUR H_Score: nan ± nan
- HARBOUR AUROC: nan ± nan
- HARBOUR A_known: nan ± nan
- HARBOUR R_novel: 0.6705 ± 0.0515

### Genus (Open-Set Evaluation)
- HARBOUR H_Score: 0.4766 ± 0.1364
- HARBOUR AUROC: 0.5145 ± 0.0696
- HARBOUR A_known: 0.3869 ± 0.1553
- HARBOUR R_novel: 0.6908 ± 0.0318

### Order (Hierarchical Generalization)
- HARBOUR H_Score: 0.5876 ± 0.0273
- HARBOUR AUROC: 0.6174 ± 0.0619
- HARBOUR A_known: 0.4996 ± 0.0379
- HARBOUR R_novel: 0.7223 ± 0.0798

### Class (Hierarchical Generalization)
- HARBOUR H_Score: 0.6758 ± 0.0372
- HARBOUR AUROC: 0.7650 ± 0.0693
- HARBOUR A_known: 0.5863 ± 0.0432
- HARBOUR R_novel: 0.8162 ± 0.1235

## 11C Paired Comparative Inference
Statistically distinguishable improvement at α=0.05 is achieved if the 95% CI of Δ excludes zero. Detailed results in `phase11_comparative_statistics.csv`.

## Statistical Caveat
> [!NOTE]
> Sequence-level bootstrap CIs quantify prediction-level uncertainty and do not constitute independent biological replicates. Seed-level variation provides the complementary measure of experimental reproducibility.
