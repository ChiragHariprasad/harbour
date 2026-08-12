# Master Implementation Plan (Phases 2 - 12C)

This master implementation plan establishes the strict methodological order, global experimental rules, and task boundaries required for executing the remaining phases of the HTRP project. 

> [!CAUTION]
> **GLOBAL EXPERIMENTAL RULE**
> The test set is used only once for final evaluation. No threshold, hyperparameter, model, DBSCAN parameter, marker weighting, feature selection, or methodological choice may be selected based on final test-set performance.

## Execution Order

The exact dependency sequence is mapped as follows:

| Phase    | Task                                         | Dependency Level   |
| -------- | -------------------------------------------- | ------------------ |
| **P1**   | Implementation freeze/audit                  | Completed          |
| **P2**   | Leakage audit                                | Completed          |
| **P2A**  | Data Reconstruction + Root-Cause Fix         | Required           |
| **P3**   | SupCon diagnosis + fix                       | Required           |
| **P4**   | Main benchmark                               | Required           |
| **P5**   | Validation-only threshold development        | Required           |
| **P6**   | Locked open-set evaluation                   | Required           |
| **P7**   | DBSCAN robustness                            | Required           |
| **P8**   | Real paired multi-marker validation          | Data-dependent     |
| **P9**   | Biological/embedding validation              | Required           |
| **P10**  | Error analysis                               | Required           |
| **P11**  | Statistical analysis                         | Required           |
| **P12**  | Reproducibility package                      | Required           |
| **P12B** | Numerical-claim provenance audit             | Required           |
| **P12C** | Manuscript consistency audit                 | Required           |

## Automated Verification Policy

For every experiment/phase result, the following automated schema validation must pass:
- `[ ]` file exists, schema correct, no NaN/Inf
- `[ ]` expected sample count, unique experiment ID
- `[ ]` seed, split, config hash, checkpoint, and dataset version recorded
- `[ ]` predictions length == labels length
- `[ ]` no train/test overlap; no test sequence in FAISS index
- `[ ]` metric ranges valid ($0 \le \text{acc/f1} \le 1$)

---

## Phase Breakdown

### Phase 2: Leakage Audit
- **Goal:** Prove strict separation of training, validation, testing, and retrieval index.
- **Tasks:**
  - Audit exact sequence duplicates ($S_{train} \cap S_{test} = \emptyset$).
  - Perform sequence-identity leakage analysis at 99%, 95%, and 90%. Repeat for train ↔ validation, validation ↔ test, and train ↔ test.
  - **Taxonomic overlap audit:** Verify train taxa $\cap$ test taxa $= \emptyset$ at intended holdout ranks.
  - **FAISS leakage audit:** Prove no test sequence or test-derived embedding exists in the reference indexes ($S_{test} \cap S_{FAISS} = \emptyset$). Validate that validation sequences are not present in test/reference index.

### Phase 2A: Root-Cause and Data Reconstruction
- **Goal:** Rebuild un-leaked datasets from the source sequence, completely free from exact duplicates and taxonomy leakage.
- **Root-Cause Analysis:**
  - **Exact sequence duplication (5,515 sequences):** `split_data.py` lacks sequence de-duplication. Identical sequences natively present in `subsampled.fasta` were simply shuffled and distributed randomly between train and test.
  - **Taxonomic Leakage (91.1% Families):** `run_hierarchical.ps1` calls `split_data.py` but omits the `--holdout_rank` argument, causing it to fall back to `-1` (no holdout). The dataset was randomly split, effectively voiding the 90/10 true open-set protocol.
- **Tasks:**
  - **Preservation:** Rename the current `assets/HTRP_cache` to `assets/HTRP_cache_LEAKED_P2`. Do not overwrite or delete existing checkpoints. Mark all metrics/checkpoints from this split as `INVALID_FOR_FINAL_PAPER = true`.
  - **Reconstruction:** Rewrite `split_data.py` to enforce strict sequence de-duplication (no identical sequences in the split pool). 
  - **Splitting:** Generate NEW immutable train, validation, and test splits directly from the original `subsampled.fasta` using `--holdout_rank 5` (Family level holdout). Ensure 0 sequence overlap and 0 taxonomic overlap at the holdout rank.
  - **Manifest Generation:** Generate a `split_manifest.json` documenting hashes, seed, holdout rank, and counts.
  - **Re-Audit:** Run the strict exact/taxonomic leakage audit AGAIN on the reconstructed dataset. Once passed, perform 99%, 95%, and 90% sequence-identity analysis.
  - **Re-Indexing:** Rebuild the FAISS indexes ONLY from the new training reference data. Phase 3 cannot begin until Phase 2A passes.

### Phase 3: SupCon Evaluation Diagnosis + Fix
- **Goal:** Determine the exact location and cause of the SupCon zero-metric evaluation failure.
- **Tasks:** 
  - Trace the full path: `training checkpoint -> checkpoint loading -> sequence preprocessing -> tokenization -> embedding generation -> FAISS/classifier -> label decoding -> y_true/y_pred -> metrics`.
  - Test independently via direct model inference, `benchmark.py`, `hierarchical_infer.py`, and `app.py`.
  - **Verification Rule:** SupCon evaluation validation must pass before any final benchmark or open-set result using SupCon is generated.

### Phase 4: Main Benchmark
- **Goal:** Comprehensive baseline comparisons.
- **Tasks:** 
  - **Design:** The 5 taxonomic splits $\times$ 5 seeds = 25 runs/method design is the default. Any reduction must be explicitly documented and approved before execution.
  - Required methods: Random Forest, SupCon.
  - Conditional methods: InfoNCE (only if reproducible checkpoint exists), Triplet (legacy only if reproducible).
  - **Closed-set metrics:** accuracy, macro precision/recall/F1, weighted-F1, balanced accuracy, MCC.

### Phase 5: Threshold Development
- **Goal:** Lock `Z_NOVELTY_THRESHOLD` and `entropy`.
- **Tasks:** 
  - Use **validation data only**.
  - The threshold-selection objective must be frozen before inspecting test-set performance. The threshold must not be selected using the test-set optimum.
  - Sweep $Z \in \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}$ and entropy candidates. Record the entire sweep (threshold, AUROC, AUPRC, F1, precision, recall, FPR@95TPR).
  - **Entropy Audit:** Determine if a single global entropy threshold is implemented in the inference path. Implement only after defining a decision rule.

### Phase 6: Locked Open-Set Evaluation
- **Goal:** Evaluate true open-set metrics using frozen thresholds.
- **Tasks:** 
  - No test-set tuning permitted.
  - Evaluate on all predefined independent taxonomic open-set splits.
  - Preserve split IDs in every output and report per-split and pooled results.
  - **Binary novelty metrics:** AUROC, AUPRC, FPR@95TPR, novelty precision/recall/F1.

### Phase 7: DBSCAN Robustness
- **Goal:** Evaluate if DBSCAN produces stable, biologically interpretable clusters among novel sequences.
- **Tasks:** 
  - Use all independently available candidate novel embeddings (target $\ge 1000$ if available; do not fabricate).
  - Metrics (No Ground Truth): Silhouette, Davies-Bouldin, noise rate, cluster size distribution, perturbation stability.
  - Metrics (If Ground Truth Exists): ARI, NMI, Purity.
  - **Cluster stability:** Bootstrap the candidate pool and calculate $Stability = ARI(C_{original}, C_{bootstrap})$.

### Phase 8: Real Paired Multi-Marker Validation
- **Goal:** Validate hierarchical weighted consensus.
- **Tasks:** 
  - **Data Dependency Check:** Inventory available datasets. If $\ge 500$ genuinely linked multi-marker biological samples exist (e.g., same specimen/voucher/accession), proceed. Do not arbitrarily pair unrelated database records. If no dataset exists, stop Phase 8 and report the gap.
  - Test conflict resolution, missing markers, and ablate the $Q \times R \times C$ formula.

### Phase 9: Biological/Embedding Validation
- **Goal:** Formal metric distance correlations.
- **Tasks:** 
  - The taxonomic-distance formula and hierarchy must be frozen before calculating the correlation. Do not choose the definition after observing which produces the strongest correlation.
  - Calculate embedding vs taxonomic distance correlation and nearest-neighbor purity.
  - **External Dataset:** Freeze the independent external marine eDNA dataset only after configurations are locked (must not have influenced prior development).

### Phase 10: Error Analysis
- **Goal:** Categorize failure modes objectively.
- **Tasks:** 
  - Extract False known, False novel, Wrong genus, Wrong family, Unresolved.
  - Provide table with Count and Rate (%). 
  - Representative examples must be selected using predefined categories (highest-confidence false known/novel, genus/family confusion, unresolved) and must include sample ID, true label, predicted label, confidence, novelty score, entropy, marker, and error category.

### Phase 11: Statistical Analysis
- **Goal:** Generate robust confidence intervals.
- **Tasks:** 
  - Bootstrap must operate on **saved predictions/scores** ($O(1000N)$), not by rerunning models.
  - Generate Descriptive statistics (Mean, SD, Median, IQR).
  - **Comparative statistics:** For paired HTRP-vs-baseline comparisons, bootstrap matched split/seed evaluation units ($\Delta_i = M_{HTRP,i} - M_{baseline,i}$) rather than independently bootstrapping methods.
  - Minimum bootstrap resamples: $B = 1,000$. Random seed for bootstrap must be recorded.

### Phase 12: Reproducibility Package
- **Goal:** Output pipeline structure for release.
- **Tasks:** 
  - Generate `Dockerfile`, `environment.yml`, and `requirements.txt`.
  - Record environment versions (Python, CUDA, PyTorch, FAISS, scikit-learn, NumPy, Pandas).
  - Generate immutable `dataset_manifest.json` (ID, source, dates, file names, SHA-256 hashes, counts, split assignments).
  - Provide a strict `reproduce_all.py` that covers: `Raw dataset -> preprocessing -> split -> model loading/training -> inference -> predictions -> statistics -> tables -> figures`.

### Phase 12B: Numerical-claim Provenance Audit
- **Goal:** Establish proven provenance for every numerical claim in the manuscript.
- **Tasks:** 
  - Trace every claim explicitly: `MANUSCRIPT CLAIM -> TABLE / FIGURE -> RESULT FILE -> EXPERIMENT ID -> PREDICTIONS / SCORES -> CHECKPOINT -> CONFIG HASH -> CODE COMMIT -> DATASET VERSION / SHA256 -> SPLIT MANIFEST -> SEED`.
  - **Rule:** If a claim cannot be traced to reproducible experiments, remove it or rerun it.

### Phase 12C: Manuscript Consistency Audit
- **Goal:** Ensure the manuscript describes exactly what was actually implemented and evaluated.
- **Tasks:**
  - Verify all textual descriptions match the frozen implementations.
  - Checklists: architecture, loss function, k-mer, markers, taxonomy ranks, thresholds, DBSCAN parameters, FAISS index, dataset sizes, metrics, figures, tables, and supplementary material all strictly align with the raw results.
