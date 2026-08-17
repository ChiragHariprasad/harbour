# Manuscript Claim Resolutions & Clarifications

This document provides the exact data and code references required to resolve the specific inconsistencies flagged for the final manuscript.

---

### 1. Jaccard Terminology: Asymmetric Containment
**Issue:** The paper uses "Jaccard similarity," but the code implementation is different.
**Data & Resolution:** 
In `app.py` (Line 114), the function is defined as:
```python
def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / len(a)
```
**Conclusion:** The code computes **Asymmetric Containment** (intersection over the length of the query $A$), not the true Jaccard Index (intersection over union). The manuscript must be updated to correctly state that the deterministic filter uses an asymmetric containment ratio to ensure short query sequences are not heavily penalized by longer reference sequences.

---

### 2. IVFFlat vs. HNSW32
**Issue:** The README and older drafts claim `IndexIVFFlat`, but the final evaluation differs.
**Data & Resolution:**
The final audited pipeline explicitly instantiates an HNSW index, not IVFFlat. 
From `build_index.py` (Line 56) and `FINAL_METHOD_CONFIG.yaml` (Line 87):
- **Index Type:** `faiss.IndexHNSWFlat`
- **Edges per node (M):** `32`
- **efConstruction:** `200`
**Conclusion:** The manuscript must state that `IndexHNSWFlat` was utilized for the final embedding evaluations to prioritize high recall. 

---

### 3. 178k QPS Claim
**Issue:** Is 178k QPS the final performance of the deployed index?
**Data & Resolution:**
The claim in `README.md` ("Achieves ~178,000 queries/sec using IndexIVFFlat") originates from an earlier scalability benchmark. Because the final pipeline was upgraded to `IndexHNSWFlat` (which trades some raw throughput for higher query accuracy and lack of training requirement), the 178k QPS figure specifically reflects the **IVFFlat structural benchmark**, not the final HNSW latency.
**Conclusion:** Contextualize the 178k QPS metric as the baseline FAISS scalability throughput tested via `IndexIVFFlat`, separate from the final HNSW implementation.

---

### 4. Thresholds: Runtime vs. Research Evaluation
**Issue:** Discrepancy between code defaults and research evaluation thresholds.
**Data & Resolution:**
- **Runtime Defaults:** `app.py` sets live serving defaults to `Z=2.0` and `C=0.6`.
- **Research Thresholds:** The audited evaluation protocol (recorded in `dataset_manifest.json` under `frozen_thresholds`) froze the statistical thresholds at **`supcon_z: 1.0`** and **`supcon_confidence: 0.5`**.
**Conclusion:** When reporting the benchmark metrics (F1, AUROC, Accuracy) in the manuscript, explicitly specify the use of the frozen evaluation thresholds ($Z=1.0, C=0.5$). The $Z=2.0$ threshold is merely a strict default for the live FastAPI inference server to minimize false positives in production.

---

### 5. Marine Case Study: 489/495 (Not 495/495)
**Issue:** Overstatement of the marine eDNA validation results.
**Data & Resolution:**
According to the final audit logs (`FINAL_PROJECT_AUDIT.md`, Line 17), the exact, verified result is that out of 495 unique environmental sequences, **exactly 489** were consistently rejected as NOVEL across all 20 tested configurations. 
**Conclusion:** The manuscript must be corrected from 495/495 to 489/495 to reflect the audited empirical data.

---

### 6. Definition of "Biological Discovery"
**Issue:** Clarifying what "discovery" means in the context of this pipeline.
**Data & Resolution:**
The pipeline's "Dark Taxa" logic is driven by the `DarkTaxaClusterer` in `app.py`, which applies DBSCAN (`eps=0.30`, `min_samples=5`) to the $L_2$-normalized embeddings of sequences rejected by the reference library.
**Conclusion:** The phrase "biological discovery" in the manuscript must be strictly defined as the detection of **reference-relative novelty** (sequences highly divergent from the known database) that form **dense, highly structured clusters in the latent space**, indicating a cohesive unseen taxonomic group. It does not imply the empirical physical confirmation of a new biological species.
