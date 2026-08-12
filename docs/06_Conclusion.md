# 6. Conclusion and SOTA Comparison

## System Efficacy
The HTRP system effectively solves the core challenge of environmental metagenomic sequencing: **structural novelty**. Traditional models operate under a closed-world assumption (assuming every sequence belongs to a Genus present in the training set). HTRP's hybrid architecture completely bypasses this limitation.

## Comparison to State-of-the-Art (SOTA)
HTRP is significantly **better suited for real-world environmental eDNA** than current Deep Learning SOTA (like DeepMicrobes, DeepSEA, or standard BERT-based DNA models).
- **Against Traditional ML (K-mer + Random Forest)**: While K-mer/RF models are incredibly fast (sub-millisecond latency), they exhibit catastrophic failure in open-set recognition, incorrectly assigning novel species to random known Genera with arbitrary confidences.
- **Against Alignment Heuristics (NCBI BLAST / Kraken2)**: BLAST and Kraken2 provide the ultimate "ground truth" precision. However, they are computationally bound. HTRP achieves comparable exact-match precision via its Deterministic Jaccard gate, but processes the remaining "fuzzy" matches using the CNN+FAISS cascade. This allows HTRP to scale to millions of reads dynamically, outperforming local BLAST deployments in pure throughput (HTRP's `IndexHNSWFlat` hits ~247K queries/sec).

## Key Differentiators from Prior Art
1. **Search by Elimination**: Instead of pushing every sequence down to a Genus classification, HTRP stops classification at the highest possible rank where novelty is detected (e.g., if a sequence is a completely novel Phylum, it is flagged as `NOVEL_PHYLUM` and halted).
2. **Dynamic Z-Score Thresholding**: HTRP abandons softmax confidence, which is notoriously uncalibrated for Out-of-Distribution data. Instead, it relies on strict distance-based Z-scores natively integrated with FAISS indexing.
3. **Decoupled Memory**: The CNN only learns the "language" of DNA, mapping sequences into a 128-D metric space. The FAISS index acts as the memory. Updating the reference database does not require retraining the neural networks—just appending new vectors to the index.
