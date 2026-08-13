# Phase 3: SupCon Evaluation Diagnosis & Redesign - Consolidated Outputs

## 1. Root Cause Analysis
We successfully diagnosed the "zero-metric" bug that plagued the SupCon evaluation in Phase 2. The issue was not a failure in the SupCon embeddings or model training, but rather a combination of two major methodological desynchronizations:

1. **Closed-Set Metrics on Open-Set Data**: In Phase 2, we successfully enforced strict taxonomic separation (`Train Taxa ∩ Test Taxa = ∅`). This meant the validation and test sets contained **100% novel sequences**. The `benchmark.py` script was attempting to compute Closed-Set accuracy on these novel sequences by matching string labels (e.g., `Microcystaceae`), which naturally returned 0.0 since the model is supposed to predict `"NOVEL"`.
2. **FAISS Index Desynchronization**: While we resplit the datasets in Phase 2, the actual FAISS indexes were never rebuilt on the new splits. Consequently, `app.py` was searching an outdated FAISS index where the distances were inherently mismatched.
3. **Z-Score Calculation Bug**: The `build_index.py` script computed empirical novelty statistics using ONLY the 1st nearest neighbor (`k=1`), while `app.py` computed the Z-score using the mean distance of the Top 10 neighbors (`k=10`). This artificially inflated the Z-score to ~15.0, forcing the model to reject everything as NOVEL regardless of confidence.

## 2. Redesign Implementation
- **Baseline Rejection Mechanism**: We updated the Classical Baseline (`MultinomialNB`) to calculate a pseudo-novelty score (`1.0 - max(predict_proba)`) and reject predictions if this score exceeded 0.5.
- **Evaluation Splitting**: We built a custom evaluation script (`test_evaluation.py`) that properly splits 500 Known samples from `train.fasta` (using LOOCV to avoid exact matches) and 500 Novel samples from `val.fasta`.
- **Target Preservation**: The redesign successfully preserves `true_taxon` without modifying biological labels and uses a binary `true_novel` flag for true open-set evaluation.

## 3. Results (Diagnosis Run)
After rebuilding the FAISS index for `family` (Seed 42) and patching the Z-score calculation to correctly use Top-1 distances (`dists[0]`), the zero-metric bug was resolved.

### Open-Set Metrics (Novel vs Known Detection)
| Model    | AUROC | AUPRC | FPR@95TPR | Novelty Precision | Novelty Recall | Novelty F1 |
|----------|-------|-------|-----------|-------------------|----------------|------------|
| SupCon   | 0.730 | 0.686 | 0.821     | 0.599             | 0.698          | 0.645      |
| Baseline | 0.675 | 0.611 | 1.000     | 0.636             | 0.014          | 0.027      |

*Note: The Baseline correctly struggles with novelty detection (Recall = 1.4%).*

### Closed-Set Metrics (Known Taxa Accuracy)
| Model    | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Balanced Acc |
|----------|----------|-----------------|--------------|----------|-------------|--------------|
| SupCon   | 0.449    | 0.411           | 0.387        | 0.391    | 0.465       | 0.412        |
| Baseline | 0.493    | 0.340           | 0.379        | 0.346    | 0.452       | 0.423        |

### Confusion Matrix
| Model    | True Class | Predicted Class | Count |
|----------|------------|-----------------|-------|
| SupCon   | Known      | Known           | 271   |
| SupCon   | Known      | Novel           | 232   |
| SupCon   | Novel      | Known           | 150   |
| SupCon   | Novel      | Novel           | 347   |

## 4. Next Steps
The pipeline is now mathematically sound and correctly distinguishes between Closed-Set and Open-Set evaluation. We are ready to proceed to Phase 4 (Main Benchmark). All necessary CSV outputs (`prediction_audit.csv`, `open_set_results.csv`, `closed_set_results.csv`, `confusion_matrix.csv`) have been saved to this directory.
