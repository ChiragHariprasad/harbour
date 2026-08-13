# Master Task Tracker (Phases 2-12C)

## Phase 2: Leakage Audit
- `[x]` Audit exact sequence duplicates ($S_{train} \cap S_{test} = \emptyset$)
- `[/]` Sequence-identity leakage (99%, 95%, 90%) - *Skipped due to massive exact/taxonomic leakage*
- `[x]` Taxonomic overlap audit
- `[x]` FAISS leakage audit ($S_{test} \cap S_{FAISS} = \emptyset$)

## Phase 2A: Data Reconstruction
- `[x]` Preserve current leaked cache to `HTRP_cache_LEAKED_P2`
- `[x]` Mark leaked metrics and checkpoints as INVALID
- `[x]` Rewrite `split_data.py` with strict sequence deduplication
- `[x]` Generate new train/val/test splits (Family holdout, 0 sequence overlap)
- `[x]` Generate `split_manifest.json`
- `[x]` Pass exact/taxonomic re-audit on new dataset
- `[x]` Run sequence-identity leakage analysis (99, 95, 90) on new splits
- `[x]` Rebuild FAISS indexes from new train data

## Phase 2A.1: Dataset Acceptance Audit
- `[x]` Independent verification of exact/taxonomic overlap
- `[x]` Independent verification of FAISS indexes
- `[x]` Sequence similarity accounting
- `[x]` Validate experimental design (5x4 seeds)
- `[x]` Generate json/csv reports and dataset manifest

## Phase 3: SupCon Evaluation Diagnosis + Redesign
- `[x]` Implement Baseline novelty/rejection mechanism (`src/baseline.py`)
- `[x]` Trace SupCon inference pipeline and novelty calculation
- `[x]` Write `evaluate_pipeline.py` with separate closed/open subset processing
- `[x]` Write unit tests for Known and Novel targets
- `[x]` Execute evaluation on validation sets and save `closed_set_results.csv`, `open_set_results.csv`, `prediction_audit.csv`, `confusion_matrix.csv`

## Phase 4: Main Benchmark
- `[x]` Run 5 taxonomic splits $\times$ 5 seeds (Parallel multi-core processing complete)
- `[x]` Evaluate Random Forest & SupCon
- `[x]` Calculate closed-set metrics

## Phase 4B: Baseline Correction & Rebenchmark
- `[x]` Root-cause anomaly (labels forced to Family)
- `[x]` Correct Baseline to use rank-specific target labels
- `[x]` Translate SupCon FAISS output to target labels
- `[x]` Re-run full 20 configurations and verify $\Delta$

## Phase 5: Threshold Development
- `[x]` 1. Generate Raw Novelty Scores (`phase5_generate_scores.py`)
  - `[x]` Rigorously assert validation dataset holdout and taxonomic logic
  - `[x]` Predict with SupCon (Z, confidence) and Baseline (max posterior)
  - `[x]` Output `phase5_raw_scores.csv` with $\mu$ / $\sigma$ provenance
- `[x]` 2. Threshold Sweep (`phase5_threshold_sweep.py`)
  - `[x]` Sweep $Z \in \{0.5...4.0\}$ and $Conf \in \{0.1...1.0\}$
  - `[x]` Sweep baseline posterior $\tau$
  - `[x]` Calculate AUROC, AUPRC, H-score per config
- `[x]` 3. Threshold Selection & Bootstrap
  - `[x]` Optimize mean validation $H_{global}$ subject to $A_{os} \ge 0.8 \times A_{cs}$
  - `[x]` Calculate 1,000 bootstrap CI for chosen threshold
  - `[x]` Run degenerate controls
- `[x]` 4. Freeze & Output
  - `[x]` Save `threshold_selection.json`
  - `[x]` Move true open-set metrics on frozen thresholds
- `[ ]` Report per-split and pooled results

## Phase 6: Final Open-Set Evaluation
- `[x]` 1. Phase 6A: Sequential Freeze Protocol
  - `[x]` Calculate optimal baseline threshold from Phase 5 scores
  - `[x]` Save `baseline_threshold_selection.json`
  - `[x]` Save `phase6_freeze_manifest.json`
- `[x]` 2. Phase 6B: Rank-Aware Integrity Audit
  - `[x]` Assert sequence ID holdouts ($train\_fit \cap test = \emptyset$, etc.)
  - `[x]` Implement rank-specific taxonomic assertions (0 for Family/Genus)
- `[x]` 3. Phase 6C: Rank-Specific Novelty Ground Truth
  - `[x]` Calculate `true_status_at_rank`
  - `[x]` Generate raw continuous scores for all 20 configurations
- `[x]` 4. Phase 6D: Scientific Metrics & Composition
  - `[x]` Output `phase6_dataset_composition.csv`
  - `[x]` Compute 14-metric suite with 1,000 bootstrap CIs
  - `[x]` Calculate Mean $\pm$ SD across 5 seeds

## Phase 7: DBSCAN Robustness
- `[x]` 1. Phase 7A: Development (Validation Set)
  - `[x]` Create `phase7A_dbscan_dev.py`
  - `[x]` Run independent grid searches (eps: 0.1-0.5, min_samples: 3-15) for all 20 combinations
  - `[x]` Implement Lexicographic metric ranking (`Stability` -> `Silhouette` -> `-Davies-Bouldin`)
  - `[x]` Calculate intersection stability over 30 bootstraps
  - `[x]` Dump frozen unsupervised parameters to `dbscan_hyperparameters.json`
- `[x]` 2. Phase 7B: Final Evaluation (Test Set)
  - `[x]` Create `phase7B_dbscan_eval.py`
  - `[x]` Load frozen DBSCAN parameters
  - `[x]` Re-run NOVEL rule on Phase 6 Open-Set candidates
  - `[x]` Compute ARI, NMI, Purity against Ground Truth taxonomy
  - `[x]` Calculate 1,000 empirical Null-Model permutations
  - `[x]` Save `phase7_final_metrics.csv` and generate `Phase7_FINAL_REPORT.md`

## Phase 8: Real Paired Multi-Marker Validation
- [x] BLOCKED: Required paired multi-marker metadata unavailable. Do not attempt to construct artificial 16S/CO1 pairs.

## Phase 9: Biological/Embedding Validation
- [x] **Phase 9A: Protocol Freeze**
    - [x] Create `phase9_freeze_manifest.json` documenting taxonomy order, missing-taxonomy logic, pair-sampling algorithms, and NN exclusion rule.
- [x] **Phase 9B: Embedding-Taxonomy Correlation**
    - [x] Create script for embedding vs. taxonomy distance calculation.
    - [x] Perform Balanced pair sampling.
    - [x] Perform Natural-distribution pair sampling (Sensitivity Analysis).
    - [x] Output `phase9_pair_samples.csv`, `phase9_embedding_taxonomy_correlations.csv`, and `phase9_correlation_bootstrap_cis.csv`.
- [x] **Phase 9C: k-NN Taxonomic Purity**
    - [x] Create script to find 10-NN (strict self-match exclusion, valid reference population).
    - [x] Compute proportion sharing Class, Order, Family, Genus.
    - [x] Output `phase9_nn_purity_predictions.csv` (raw audit trail) and `phase9_nn_purity_summary.csv` (aggregate mean±SD).
- [x] **Phase 9D: Baseline Comparison**
    - [x] Run identical correlation and purity analyses for reproducible baseline representations (if applicable) and output `phase9_baseline_comparison.csv`.
- [x] **Phase 9E: Independent Marine eDNA Case Study**
    - [x] Zero-shot prediction and clustering on `case_study_marine.fasta`.
    - [x] Output `phase9_marine_case_study.csv`.
- [x] Generate final `Phase9_Biological_Embedding_Report.md`.

## Phase 10: Error Analysis
- `[ ]` Categorize False Known, False Novel, Wrong Genus, Wrong Family, Unresolved
- `[ ]` Output table of error frequencies
- `[ ]` Extract structured representative examples (highest conf, homonyms, etc.)

## Phase 11: Statistical Analysis
- `[ ]` Bootstrap saved predictions ($B=1000$)
- `[ ]` Calculate Descriptive statistics & 95% CIs
- `[ ]` Calculate Comparative paired statistics

## Phase 12: Reproducibility Package
- `[ ]` Generate `Dockerfile`, `environment.yml`, `requirements.txt`
- `[ ]` Generate `dataset_manifest.json`
- `[ ]` Provide `reproduce_all.py`

## Phase 12B: Numerical-claim Provenance Audit
- `[ ]` Trace all manuscript numerical claims

## Phase 12C: Manuscript Consistency Audit
- `[ ]` Verify text descriptions match implementation
