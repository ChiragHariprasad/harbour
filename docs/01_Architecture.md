# 1. System Architecture

## Architecture Diagram

```mermaid
graph TD
    A[Input Sequence (FASTA)] --> B{Deterministic Matcher\n21-mer Jaccard}
    B -- Match ≥ 0.95 --> C[Known Species]
    B -- Match < 0.95 --> D[Hierarchical ML Cascade]
    
    subgraph Hierarchical ML Cascade [Search by Elimination Strategy]
        D --> E{Phylum Model}
        E -- Z > 1.5 --> F[NOVEL PHYLUM]
        E -- Z ≤ 1.5 --> G{Class Model}
        G -- Z > 1.5 --> H[NOVEL CLASS]
        G -- Z ≤ 1.5 --> I{Order Model}
        I -- Z > 1.5 --> J[NOVEL ORDER]
        I -- Z ≤ 1.5 --> K{Family Model}
        K -- Z > 1.5 --> L[NOVEL FAMILY]
        K -- Z ≤ 1.5 --> M{Genus Model}
        M -- Z > 1.5 --> N[NOVEL GENUS]
        M -- Z ≤ 1.5 --> O[Known Genus]
    end
```

## Explanation and Reasoning

The HTRP (Harbour Taxonomic Recovery Pipeline) utilizes a hybrid architecture specifically designed to solve the **Open-Set Recognition (OSR)** problem in metagenomics, where environmental sequences often belong to organisms completely absent from training databases.

### 1. Deterministic Matcher (Fast Path)
The pipeline begins with a highly conservative deterministic filter utilizing a 21-mer Jaccard similarity index. 
**Reasoning**: Deep learning models are computationally expensive and prone to overconfidence on exact matches if not perfectly calibrated. By routing sequences with $\ge 95\%$ Jaccard similarity directly to a "Known Species" classification, we guarantee 100% precision for trivially known sequences, drastically reducing the computational burden on the ML cascade.

### 2. Hierarchical ML Cascade
If a sequence fails the deterministic check, it enters a 5-stage Machine Learning Cascade (Phylum $\rightarrow$ Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus). Each stage consists of an independent 1D-CNN encoder and a FAISS similarity index.
**Reasoning**: Traditional "flat" classifiers force every sequence into a specific Genus/Species bucket, leading to severe hallucinations when presented with novel taxa. We implement a **Search by Elimination** ideology. The pipeline checks novelty at the highest taxonomic rank first (Phylum). If a sequence is deemed novel at the Phylum level, classification halts immediately, preventing the downstream Class/Order/Family models from making forced, erroneous predictions. 

### 3. Z-Score Anomaly Gating
At each ML node, the sequence is embedded into a 128-dimensional vector. Its distance to the nearest $K$ known neighbors in the FAISS index is calculated. We use dynamic Z-score thresholding ($Z > 1.5$) based on the index's empirical mean and standard deviation.
**Reasoning**: Instead of relying on traditional softmax confidence thresholds (which suffer from overconfidence in OOD scenarios), distance-based Z-scores directly quantify structural deviation from the training manifold. A threshold of 1.5 standard deviations was empirically selected to perfectly balance Novelty Recall and the False Positive Rate at 95% True Positive Rate (FPR@95TPR).
