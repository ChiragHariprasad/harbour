# 3. Models and Algorithms

The HTRP system intentionally avoids relying on a single monolithic architecture, instead utilizing a suite of specialized models at different stages of the pipeline.

## 1. 1D Convolutional Neural Networks (Deep Learning)
At the core of the ML Cascade are 5 independent PyTorch 1D-CNN models, one for each taxonomic rank (Phylum through Genus).
- **Architecture**: A lightweight embedding network consisting of an embedding layer, followed by a 1D Convolutional layer (kernel size=3), ReLU activation, Global Average Pooling, and a Linear projection layer that outputs a 128-dimensional latent vector.
- **Why CNN over LLMs/Transformers?**: DNA sequences (especially amplicon fragments) are long (150-500bp) and rely heavily on local motif patterns (k-mers). Transformers scale quadratically ($O(N^2)$) with sequence length, making them computationally prohibitive for real-time processing of millions of metagenomic reads. A 1D-CNN extracts local features with linear complexity ($O(N)$), enabling microsecond latency per sequence on standard CPUs while maintaining robust embedding quality.

## 2. Supervised Contrastive Learning (SupCon)
The CNNs are not trained using standard Cross-Entropy loss. They are trained using **Supervised Contrastive Loss**.
- **Reasoning**: Cross-Entropy forces the network to learn a rigid hyperplane separating predefined classes (closed-set). SupCon forces the network to learn a generalized *metric space*—pulling sequences of the same taxon closer together and pushing different taxa apart. This is structurally required for Open-Set Recognition, as it clusters knowns tightly, leaving novel sequences isolated in the latent space where they can be flagged by distance thresholds.

## 3. FAISS Vector Search (Machine Learning)
Once the CNN maps a sequence into the 128-D space, we use Meta's **FAISS (Facebook AI Similarity Search)** to perform K-Nearest Neighbors (KNN).
- **Reasoning**: FAISS is the industry standard for extreme-scale vector retrieval. Our architecture utilizes `IndexHNSWFlat` and `IndexIVFFlat`, scaling to millions of sequences with sub-millisecond latency. This decouples the "memory" of the database from the "logic" of the neural network. To add a newly discovered species to the system, researchers simply add its embedding to the FAISS index—no expensive CNN retraining required.

## 4. Baselines (Random Forest & BLAST)
To validate the architecture, the system is actively benchmarked against two alternatives:
- **K-mer + Random Forest**: A traditional ML baseline using `CountVectorizer` and scikit-learn's `RandomForestClassifier`. While effective for small closed-set tasks, it struggles to scale to thousands of taxa and fails to produce calibrated probabilities for open-set novelty detection.
- **NCBI BLAST**: The gold standard heuristic alignment algorithm. While BLAST provides excellent precision, its local alignment algorithm is too slow for real-time analysis of massive eDNA dumps. The HTRP cascade achieves comparable novelty detection at a fraction of the computational cost.
