# FINAL PROJECT AUDIT: HARBOUR Analytics Pipeline

## Project Status: COMPLETED

This document certifies the successful completion of the HARBOUR analytical evaluation pipeline (Phases 5 through 12C).

### 1. Robustness & Entropy
- **Frozen Framework**: The system strictly avoided dynamic entropy injections.
- **DBSCAN Freeze**: Parameters ($\epsilon, MinPts$) were frozen prior to Phase 7 and were not tuned on the test set.
- **Embedding Metric**: Strict use of cosine distance was enforced throughout all nearest-neighbor and clustering searches.

### 2. Statistical Discipline
- **Resampling**: Paired sequence-level bootstrap ($B=1,000$) implemented properly. $\Delta$ was mathematically checked per iteration.
- **Null Models**: Label-permutation controls established expected baseline bounds for taxonomic clustering, strictly averting "proof" claims in favor of empirical deviation from the null.
- **Failures & Anomalies Captured**:
  - Captured undefined metrics (`NaN`) dynamically rather than failing silently when ground truth lacked expected labels (e.g. true KNOWNS in open-set evaluations).
  - Explicitly caught a transcription risk: **Marine case study was evaluated over 495 unique sequences; exactly 489 were consistently NOVEL across 20 configurations, not 495.**

### 3. Reproducibility Guarantee
- Phases 1-4 (Upstream Training): Represented strictly by frozen provenance artifacts.
- Phases 5-11 (Evaluation): Fully reproducible via `reproduce_all.py`.
- Phase 12 (Audit): Concluded with exact numerical and methodological compliance checks.

The computational codebase and the manuscript claims are now completely aligned.
