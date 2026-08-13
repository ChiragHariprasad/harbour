# Phase 4.1 Audit Report

## Overall Status: PASS

## 1. Results Integrity
- 40 exact configuration rows verified.
- Cartesian structure strictly validated.
- 0 NaN or missing predictions found.

## 2. Data Leakage & Provenance
- All 20 FAISS indices mathematically verified to have zero intersection with `train_eval`.
- strict subset rule `taxa(train_eval) <= taxa(train_fit)` passed for all configurations.
- Checkpoint provenance hashes perfectly match.

## 3. Metric Recalculation
- All 40 metrics recalculated independently using `sklearn.metrics`.
- Maximum absolute discrepancy is strictly `<= 1e-10`.

## 4. Baseline Audit Findings
- Baseline predictions for seed 42 are IDENTICAL for ranks class and order
- Baseline predictions for seed 42 are IDENTICAL for ranks class and family
- Baseline predictions for seed 42 are IDENTICAL for ranks class and genus
- Baseline predictions for seed 42 are IDENTICAL for ranks family and order
- Baseline predictions for seed 42 are IDENTICAL for ranks family and genus
- Baseline predictions for seed 42 are IDENTICAL for ranks genus and order
- Baseline predictions for seed 43 are IDENTICAL for ranks class and order
- Baseline predictions for seed 43 are IDENTICAL for ranks class and family
- Baseline predictions for seed 43 are IDENTICAL for ranks class and genus
- Baseline predictions for seed 43 are IDENTICAL for ranks family and order
- Baseline predictions for seed 43 are IDENTICAL for ranks family and genus
- Baseline predictions for seed 43 are IDENTICAL for ranks genus and order
- Baseline predictions for seed 44 are IDENTICAL for ranks class and order
- Baseline predictions for seed 44 are IDENTICAL for ranks class and family
- Baseline predictions for seed 44 are IDENTICAL for ranks class and genus
- Baseline predictions for seed 44 are IDENTICAL for ranks family and order
- Baseline predictions for seed 44 are IDENTICAL for ranks family and genus
- Baseline predictions for seed 44 are IDENTICAL for ranks genus and order
- Baseline predictions for seed 45 are IDENTICAL for ranks class and order
- Baseline predictions for seed 45 are IDENTICAL for ranks class and family
- Baseline predictions for seed 45 are IDENTICAL for ranks class and genus
- Baseline predictions for seed 45 are IDENTICAL for ranks family and order
- Baseline predictions for seed 45 are IDENTICAL for ranks family and genus
- Baseline predictions for seed 45 are IDENTICAL for ranks genus and order
- Baseline predictions for seed 46 are IDENTICAL for ranks class and order
- Baseline predictions for seed 46 are IDENTICAL for ranks class and family
- Baseline predictions for seed 46 are IDENTICAL for ranks class and genus
- Baseline predictions for seed 46 are IDENTICAL for ranks family and order
- Baseline predictions for seed 46 are IDENTICAL for ranks family and genus
- Baseline predictions for seed 46 are IDENTICAL for ranks genus and order
- Baseline actual model class: MultinomialNB
