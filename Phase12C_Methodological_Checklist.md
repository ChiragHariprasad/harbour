# Phase 12C: Manuscript Consistency Audit Checklist

The following items must be verified in the manuscript text to ensure perfect consistency with the implemented pipeline:

## Phase 6 (Terminology)
- [ ] Distinguish the "final open-set evaluation protocol" from a generic "test set".
- [ ] Preserve the distinction between true Family-level novelty (open-set) and higher-level hierarchical generalization (Order/Class).

## Phase 7 (Statistical Interpretation)
- [ ] The permutation result supports rejection of the specified label-permutation null.
- [ ] Ensure wording avoids claiming $p<0.001$ is "proof" that clusters are biologically "inherently" meaningful.
- [ ] The ARI/NMI/Purity and stability results provide evidence of taxonomic alignment; the permutation test establishes that the observed alignment exceeds the specified random-label null.

## Phase 8 (Missing Paired Metadata)
- [ ] Explicitly state that paired multi-marker validation was not performed because the required linked biological metadata were unavailable.
- [ ] Do not describe Phase 8 as a negative experimental result.

## Phase 9 (Marine Case Study)
- [ ] Clarify that the "NOVEL" classification is an observed model output relative to HARBOUR's reference/training database.
- [ ] It should NOT be presented as absolute proof that all marine eDNA is biologically novel in nature.
- [ ] Note: The audit script `phase12B_numerical_audit.py` caught that exactly 489 out of 495 sequences were consistently classified as NOVEL across all configurations (rather than 495/495). Update the manuscript numerical claim accordingly.

## Model Boundaries & Thresholds
- [ ] State explicitly that HARBOUR's novelty boundary is fixed at $Z \ge 1.0$ OR $Conf < 0.5$.
- [ ] State that the embedding distance metric used in DBSCAN and k-NN was $1 - \cos(\theta)$ (cosine distance).

## Phase 11 (Significance Testing)
- [ ] State that sequence-level uncertainty is bounded by a 1,000-iteration paired bootstrap.
- [ ] Mention that $\Delta$ (HARBOUR - Baseline) is calculated directly within each bootstrap replicate.
