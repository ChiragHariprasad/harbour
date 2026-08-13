# Phase 4.1: Statistical & Integrity Audit Validation Gate

## Goal Description
Conduct a rigorous integrity, leakage, and statistical audit on all Phase 4 outputs. This serves as a critical decision gate before progressing to Phase 5. We will independently verify all calculated metrics, ensure zero data leakage between splits, explain the baseline behavior, and securely organize all artifacts into structured phase directories.

## Proposed Changes

### 1. `phase4_1_audit.py` Script Creation
We will write a comprehensive Python script that executes the 11-step audit automatically:

- **Results Integrity**: Use `pandas` to load `Phase4_Main_Benchmark_Results.csv` and `phase4_predictions.csv`. Verify exactly 40 config rows, no missing predictions, and no NaN/Inf values.
- **Data Leakage & Provenance**: Load `train_fit.fasta`, `train_eval.fasta`, and FAISS indices for all 20 configs. Use `set` intersections to mathematically prove $S_{fit} \cap S_{eval} = \emptyset$ and verify taxa subsets. Cross-reference `train_metrics.json` timestamps against the `embedding.pt` modification times to verify provenance.
- **Class Distribution**: Calculate min/median/max sequence representations for every holdout rank and save to `phase4_class_distribution.csv`.
- **Metric Audit**: Independently recalculate Accuracy, Macro Precision/Recall/F1, Weighted F1, Balanced Accuracy, and MCC using `sklearn.metrics`. Assert the difference from the main benchmark CSV is strictly $\le 1\times 10^{-10}$. Save to `phase4_metric_recalculation.csv`.
- **Baseline Audit**: Investigate the exact 53.44% accuracy anomaly of `MultinomialNB`. The script will verify if identical target sequences or identical label distributions are causing this plateau across ranks, and ensure it is strictly a `MultinomialNB` model being invoked.
- **HARBOUR & Seed Variability**: Compute mean, SD, 95% CI, min, and max across the 5 seeds (unpooled) for both HARBOUR and Baseline. Save to `phase4_seed_statistics.csv`.
- **Confusion Matrices**: Dynamically reconstruct the confusion matrices from the prediction CSV and assert equality with the saved matrices in `phase4_confusion_matrices/`.
- **Effect Size**: Calculate paired $\Delta = \text{HARBOUR} - \text{baseline}$ and output to `phase4_effect_sizes.csv`.
- **Audit Report**: Aggregate the final PASS/FAIL status and explanations into `phase4_audit_report.md`.

### 2. Artifact Reorganization
Once the audit completes and passes, we will programmatically:
- Create `e:\HTRP\master Q1 details\Phase_4_Benchmark`
- Create `e:\HTRP\master Q1 details\Phase_4.1_Audit`
- Move all Phase 4 artifacts (manifests, reports, CSVs, and confusion matrices) into the Phase 4 directory.
- Move the 5 generated Phase 4.1 outputs into the Phase 4.1 directory.
- *Note: We will implement this phase-wise folder structure convention for all future phases.*

## User Review Required

> [!CAUTION]
> **Decision Gate**
> If `phase4_1_audit.py` fails any of the strict integrity checks (e.g. data leakage is detected, or metric discrepancies $> 10^{-10}$ exist), the script will intentionally halt with a `FAIL` status. I will stop execution, present you with the failure logs, and await your direction on how to fix the underlying issue before we touch Phase 5.

## Verification Plan

### Automated Tests
- Run `python phase4_1_audit.py`.
- Verify the script prints `PASS -> proceed to Phase 5` without assertion errors.
- Verify that `master Q1 details` contains the newly populated `Phase_4_Benchmark` and `Phase_4.1_Audit` directories with the corresponding files cleanly moved.
