# Phase 2 Leakage Audit Summary

I executed the strict leakage audit on the current `16S` train and test datasets across the four main taxonomic ranks (`class`, `order`, `family`, `genus`). The results demonstrate a **catastrophic breakdown of the intended 90/10 taxonomic hold-out protocol**.

> [!CAUTION]
> **Massive Dataset Leakage Detected**
> The splits currently saved in `assets/HTRP_cache` are overwhelmingly leaked. Test sequences and test taxa are almost entirely present in the training set.

## 1. Exact Sequence Duplicates ($S_{train} \cap S_{test}$)
Across every single rank, there are exactly **5,515 identical sequences** present in both the training set and the test set. 
- Because exact leakage was immediately observed at this scale, the 99%/95%/90% similarity checks were aborted as they would redundantly prove the same fatal flaw.

## 2. Taxonomic Overlap (Train Taxa $\cap$ Test Taxa)
The `02_Dataset.md` document explicitly claimed that *entire lineages were deterministically excluded from training and placed exclusively in the test set* to test true open-set performance. **This is false for the current dataset.**

| Rank   | Train Taxa | Test Taxa | Overlapping Taxa | % of Test Taxa Leaked |
|--------|------------|-----------|------------------|-----------------------|
| Class  | 303        | 258       | **254**          | **98.4%**             |
| Order  | 1,225      | 997       | **965**          | **96.7%**             |
| Family | 4,862      | 3,162     | **2,881**        | **91.1%**             |
| Genus  | 15,932     | 7,394     | **5,595**        | **75.6%**             |

If the model achieved high AUROC for novelty detection on this test set, it is because it already memorized 91% of the "novel" families during training.

## 3. FAISS Index Leakage
- **Status: CLEAR.**
- I verified that none of the test sequence IDs are embedded directly into the final `ids.npy` mapping arrays for the FAISS index at any rank.

---

### Conclusion
The current `train.fasta` and `test.fasta` files in `assets/HTRP_cache/16S` cannot be used for Phase 3 (Benchmark) or Phase 6 (Open-Set Evaluation). Any SupCon model trained on this data is permanently compromised for open-set evaluation because it has already seen the vast majority of the test taxa and exactly 5,515 of the test sequences.
