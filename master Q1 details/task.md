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
- `[ ]` Sweep $Z \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$ and entropy (validation only)
- `[ ]` Implement and define global entropy rule

## Phase 6: Locked Open-Set Evaluation
- `[ ]` Evaluate true open-set metrics on frozen thresholds
- `[ ]` Report per-split and pooled results

## Phase 7: DBSCAN Robustness
- `[ ]` Collect novel candidates ($\ge 1000$ if available)
- `[ ]` Calculate no-ground-truth metrics & stability
- `[ ]` Calculate ground-truth metrics (if applicable)

## Phase 8: Real Paired Multi-Marker Validation
- `[ ]` Inventory datasets for $\ge 500$ real paired samples
- `[ ]` Run conflict and missing marker tests

## Phase 9: Biological/Embedding Validation
- `[ ]` Define taxonomic-distance formula
- `[ ]` Calculate correlation and nearest-neighbor purity
- `[ ]` Evaluate independent external dataset

## Phase 10: Error Analysis
- `[ ]` Extract False known/novel, genus/family confusion, unresolved
- `[ ]` Select representative examples by category

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
