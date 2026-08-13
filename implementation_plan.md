# Phase 4: Main Benchmark

## Goal Description
Generate the primary closed-set benchmark across all 20 configurations (4 taxonomic ranks $\times$ 5 predefined seeds) using the frozen methodology.

## Proposed Changes

### 1. Stratified Data Splitting for Closed-Set Evaluation
We will create a script to deterministically split each configuration's `train.fasta` into `train_fit.fasta` and `train_eval.fasta` to ensure a mathematically valid closed-set evaluation.
- **Stratification Rule**: The split will be grouped by the relevant taxonomic holdout rank. 
- **Minimum Representation**: We will ensure that *every* taxon selected for the evaluation set is strictly a subset of the taxa present in `train_fit.fasta` (`taxa(train_eval) ⊆ taxa(train_fit)`).
- **Sampling Strategy**: We will perform proportional deterministic stratified sampling (using `train_test_split` with `stratify=taxa`) and cap the `train_eval` size at a maximum of 2,000 sequences. If stratification yields fewer than 2,000 valid sequences, we will use the exact yielded amount.
- **Reporting**: We will record $N_{train\_fit}$, $N_{train\_eval}$, $N_{taxa\_train\_fit}$, $N_{taxa\_train\_eval}$, and min/median/max sequences per taxon.

### 2. Model Retraining (Fixing the Provenance Leakage)
As identified, the current HARBOUR checkpoint (`embedding.pt`) was trained on the pre-Phase 2 dataset. Evaluating `train_eval.fasta` against this old model would be a data leak.
- **HARBOUR ML**: We will programmatically execute `train.py` for all 20 configurations, training strictly on the newly generated `train_fit.fasta`.
- **Classical Baseline**: We will train the `ClassicalBaseline` exclusively on `train_fit.fasta`.
*(Note: I inspected `src/baseline.py` and it explicitly implements a Naive Bayes classifier, not Random Forest. I will accurately refer to it as `ClassicalBaseline (MultinomialNB)` from here on out).*

### 3. FAISS Index Rebuild
- Following the retraining of the ML models, we will rebuild all 20 FAISS indexes using `build_index.py` against `train_fit.fasta`.
- We will verify index vector count, ID uniqueness, and ensure no evaluation sequences leak into the index.

### 4. Evaluation Script & Metrics
- We will evaluate both HARBOUR and the ClassicalBaseline using the identical `train_eval.fasta` IDs.
- Metrics calculated: Accuracy, Macro Precision/Recall/F1, Weighted F1, Balanced Accuracy, MCC.
- We will execute the closed-set metrics only; open-set and threshold optimization are strictly deferred to Phases 5/6.

### 5. Outputs & Provenance
We will produce the following artifacts in the output directory:
- `phase4_predictions.csv` (Per-sequence predictions preserving experiment ID, rank, seed, true/predicted taxon, confidence, checkpoint ID/hash, config hash, and manifest hashes)
- `Phase4_Main_Benchmark_Results.csv` (Aggregate results per model $\times$ experiment)
- `phase4_confusion_matrices/` (Confusion matrix data for each experiment)
- `phase4_dataset_manifest.json`
- `phase4_validation_report.json`

## User Review Required

> [!WARNING]
> **Computational Runtime for Retraining**
> Retraining the HARBOUR ML model (SupCon) from scratch across 20 configurations would sequentially take approximately **2.8 hours** in total. To drastically reduce this time as requested, I will implement a **multi-core parallel processing orchestrator** (using Python's `multiprocessing` or `concurrent.futures`) to execute the splits, model training, and FAISS indexing for multiple configurations concurrently across available CPU cores. This will execute in the background and should cut the time down to a fraction of the sequential estimate.

## Verification Plan
### Automated Tests
- Verify `prediction IDs == train_eval IDs` and are unique.
- Verify `len(y_true) == len(y_pred)`.
- Verify no NaN/Inf confidence values.
- Verify `taxa(train_eval) ⊆ taxa(train_fit)`.
- Verify NO `train_eval` sequence occurs in `train_fit` or the FAISS index.
- Verify exact checkpoint/config association.
