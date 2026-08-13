# Phase 5: Optimal Threshold Selection

This report details the final threshold selection procedure and bootstrap uncertainty metrics for the SUPCON model across all 20 benchmark configurations, using the held-out pure hybrid validation dataset.

## Final Selection Criterion
We swept a robust parameter grid ($Z \in \{0.5, \dots, 4.0\}$ and $Conf \in \{0.1, \dots, 1.0\}$). For every $(Z, Conf)$ pair, we evaluated all 20 configurations.

The objective was to maximize the Global H-Score (the unweighted mean of $H_{r,s}$ across the 20 configurations) subject to a rigorous known-accuracy retention constraint: $A_{os} \ge 0.8 \times A_{cs}$. 

## Optimal Operating Point
The optimal hyperparameters that maximized global H-Score while strictly passing the retention constraint are:
- **Z-Score Threshold**: `1.0`
- **Confidence Threshold**: `0.5`

*(If a sequence's $Z \ge 1.0$ OR $Conf < 0.5$, it is flagged as NOVEL).*

## Bootstrap Uncertainty (1,000 Resamples)
We performed 1,000 bootstrap iterations over the validation set at the optimal threshold to calculate 95% Confidence Intervals.

| Metric | Mean | 95% CI (Lower) | 95% CI (Upper) |
|---|---|---|---|
| **H-Score** | 0.6462 | 0.6431 | 0.6492 |
| **Known Accuracy** | 0.6154 | 0.6108 | 0.6199 |
| **Novel Recall** | 0.6804 | 0.6766 | 0.6841 |

## Conclusion
The strict validation process has definitively locked in $Z = 1.0$ and $Conf = 0.5$. The baseline and models had no access to the final `test.fasta` files, which remain mathematically pristine for Phase 6.
