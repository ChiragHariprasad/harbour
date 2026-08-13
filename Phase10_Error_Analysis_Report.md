# Phase 10: Error Analysis Report

## 1. Sequence Integrity Audit
PASS: 20 configurations, no duplicates, valid scores.

## 2. Open-Set Detection Failures (Layer A)
### HARBOUR
- False Known Rate: 31.35%
- False Novel Rate: 39.16%

## 3. Taxonomic Classification Failures (Layer B)
- Wrong Taxon Rate (among Accepted Known): 18.29%

## 4. Decision-Boundary / Error-Severity
See `phase10_error_boundary_analysis_Z.csv` and `phase10_error_boundary_analysis_Conf.csv`.

## 5. HARBOUR vs MultinomialNB Failure-Mode Comparison
| Metric            |      HARBOUR |     Baseline |
|:------------------|-------------:|-------------:|
| Total_Novel       | 61718        | 61718        |
| Total_Known       | 55078        | 55078        |
| FK                | 19351        | 28923        |
| FN                | 21568        |  9093        |
| FK_Rate           |     0.313539 |     0.468632 |
| FN_Rate           |     0.39159  |     0.165093 |
| Known_Known_Valid | 27167        | 38204        |
| Wrong_Taxon       |  4970        | 11442        |
| Wrong_Taxon_Rate  |     0.182943 |     0.299497 |