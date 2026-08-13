# Phase 6.1 — Final Statistical & Integrity Audit

## 1. Frozen Provenance Audit
- [PASS] test_used_for_selection = False
- [PASS] SupCon Z threshold frozen at: 1.0
- [PASS] SupCon Conf threshold frozen at: 0.5
- [PASS] Baseline threshold frozen at: 0.99
- [PASS] Dataset Hash verified: 05543cbba34a31e00004e6d3e12202ef9086a2b4b89916bb0deb7cabafcc5a2d

## 2. Dataset Composition & Biology Audit
- [PASS] Exactly 20 configurations evaluated (4 ranks x 5 seeds).
- [PASS] Class overlap explicitly recorded as Hierarchical Generalization. Overlapping taxa: 106
- [PASS] Family holdout perfectly strict. Overlap: 0.
- [PASS] Genus homonyms correctly identified. Max overlapping taxa across seeds: 9
- [PASS] Order overlap explicitly recorded as Hierarchical Generalization. Overlapping taxa: 242

## 3. Independent Metrics Recalculation
- [PASS] Verified AUROC calculated from continuous Z-score (where higher = more novel).
- [PASS] Family HARBOUR H-Score independently verified: 0.6937 == 0.6937
- [PASS] Class HARBOUR H-Score independently verified: 0.7272 == 0.7272