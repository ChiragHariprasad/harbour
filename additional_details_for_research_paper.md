# Additional Details for Research Paper

This document consolidates the exact, authoritative data points, methodologies, and specifications required for the final methods and results sections of the manuscript.

## 1. Model Implementation Details

### CNN Architecture
- Exact layer sequence
- Embedding/token vocabulary size
- k-mer size
- Convolution kernel sizes
- Number of filters/channels
- Pooling method
- Projection dimensions
- Final embedding dimension (*Note: Established as 128-dimensional L2-normalized representations*)
- Activation functions
- Normalization
- Dropout, if any

### SupCon (Supervised Contrastive Learning) Configuration
- Exact loss equation
- Temperature
- Positive-pair definition
- Negative-pair definition
- Batch construction
- Class/taxon sampling strategy
- Number of epochs
- Optimizer
- Learning rate
- Weight decay
- Scheduler
- Early stopping
- Checkpoint selection criterion

## 2. Dataset Provenance
**Requirement:** Provide exact dataset provenance to replace generic references like "SILVA and other databases."

Required details (for all 20 configurations, preferably from the final manifest/CSV):
- Exact database name
- Exact version
- Download date
- Accession information (if available)
- Raw sequence count (*Note: Original ~100,000 records*)
- Deduplicated count (*Note: Reduced to ~58,910 unique sequences; exact duplicate leakage eliminated*)
- Final usable count
- Number of taxa at every rank
- Preprocessing exclusions
- Sequence-length distribution
- Ambiguous-base filtering
- Taxonomy completeness rules

## 3. Split Methodology
**Protocol Overview:** Five seeds, four target ranks, strict family-level holdout, no exact sequence overlap, no target-rank taxonomic overlap, reconstructed FAISS indexes.

**Required Formal Specification:**
- Definition of $D = D_{train} \cup D_{val} \cup D_{test}$ and exact generation methodology.

For each rank/seed configuration:
- $N_{train}$
- $N_{validation}$
- $N_{test}$
- Number of unique taxa
- Number of held-out taxa
- Overlap counts
- Similarity leakage statistics

## 4. Baseline Specification
**Requirement:** Formally specify the MultinomialNB baseline evaluated in the final protocol (resolving older descriptions of Random Forest and BLAST).

Required details:
- Feature representation
- k-mer size
- Vectorizer
- n-gram range
- Vocabulary construction
- Smoothing parameter
- Training data
- Prediction procedure
- Novelty score
- Baseline threshold
- Hyperparameter tuning status

## 5. Metric Definitions
**Requirement:** Mathematically define every custom metric.

Require authoritative formulas for:
- H-Score
- $A_{cs}$ (Closed-set accuracy)
- $A_{os}$ (Open-set accuracy)
- Novel Recall
- Novel F1
- MCC
- Any other custom metrics

*Context: Threshold-selection objective was $\max H$ subject to $A_{os} \ge 0.8 A_{cs}$.*

## 6. Phase 4 Benchmark Results
**Requirement:** Retrieve the final audited Phase 4 benchmark results.

Require the actual tabular data across all 20 configurations for HARBOUR and the baseline:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Balanced Accuracy
- MCC

*Context: The Phase 4 audit confirmed that metrics were independently recalculated with a maximum discrepancy $\le 10^{-10}$, and configuration structure was validated.*

## 7. Phase 11 Comparative Inference
**Requirement:** Resolve the NaN reporting issue in Phase 11 family-level metrics.

- **Issue:** Family H-score/AUROC/A_known appear as `NaN ± NaN` in Phase 11, whereas Phase 6 has valid family-level results.
- **Action Needed:** Determine if this is a seed-level aggregation artifact, a metric-definition issue, a reporting bug, or a genuine missing quantity (e.g., expected class absent in a configuration). We need one authoritative statistical result for the manuscript.

## 8. Figures Needed
**Requirement:** Generate publication-quality plots based on the final audited CSVs.

Target Figure List (approximate for Q1 paper):
- **Figure 1 (HARBOUR Architecture):** Input $\to$ deterministic filtering $\to$ embedding $\to$ FAISS $\to$ confidence/Z-score $\to$ hierarchical rejection $\to$ candidate novelty.
- **Figure 2 (Experimental Design):** Leakage-controlled split: Raw data $\to$ deduplication $\to$ taxonomy grouping $\to$ train/validation/test $\to$ 20 configurations.
- **Figure 3 (Open-set Performance):** HARBOUR vs MultinomialNB across Class/Order/Family/Genus.
- **Figure 4 (Embedding-space Biological Structure):** Taxonomic distance vs embedding distance, with correlation and CI.
- **Figure 5 (Novelty-cluster Structure):** DBSCAN clustering of novel candidates, with stability.
- **Figure 6 (Error Landscape):** False-known vs false-novel errors as a function of Z-score and confidence.
- **Figure 7 (Independent Marine Case Study):** 495 sequences, showing the distribution of novelty decisions and explicitly marking the 489 consistently novel sequences.

## 9. Related Work & References
**Requirement:** Build a state-of-the-art reference package using current peer-reviewed literature.

Beyond foundational citations (BLAST, VSEARCH, RDP, SILVA, PR2, BOLD, UNITE, MitoFish, eDNA), we need recent literature on:
- eDNA taxonomic assignment
- Open-set recognition and OOD detection
- Contrastive learning
- Biological sequence representation learning
- k-mer/alignment-free methods
- Learned nearest-neighbor representations
- Taxonomic hierarchical classification
- Uncertainty/calibration
- Novelty discovery in biodiversity
- DBSCAN/unsupervised clustering
- Statistical evaluation of ecological ML

## 10. Scientific Limitations (Discussion Point)
**Requirement:** Explicitly discuss the error trade-offs to support a risk-sensitive open-set recognition argument.

Key observations:
- **HARBOUR:** False Known Rate: 31.35%, False Novel Rate: 39.16%, Wrong Taxon (among accepted known): 18.29%
- **Baseline:** False Known Rate: 46.86%, False Novel Rate: 16.51%, Wrong Taxon Rate: 29.95%
- **Insight:** HARBOUR shifts the decision boundary toward more conservative rejection of uncertain sequences. It substantially reduces false acceptance of novel sequences, at the cost of rejecting more genuinely known sequences.

## 11. Reproducibility
**Requirement:** Include a dedicated subsection highlighting reproducibility.

Key assertions to document:
- Project correctly separates frozen provenance artifacts (Phases 1–4) from reproducible steps (Phases 5–7, 9–11, 12). Phase 8 is intentionally skipped.
- The evaluation pipeline can be rerun via `reproduce_all.py` with cryptographic input-hash verification.
- Be explicitly honest that the original training environment/checkpoints are only available as provenance artifacts, which enhances credibility.
