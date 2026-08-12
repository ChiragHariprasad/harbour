# 5. Experimental Results

The hierarchical pipeline was rigorously evaluated against the `16S` test fold utilizing the strict taxonomic open-set hold-out protocol. The results below reflect the final tuned architecture ($Z=1.5$ novelty threshold, $J=0.95$ deterministic threshold).

## Main Benchmarks
| Metric | Hierarchical Pipeline | Traditional K-mer (RF) Baseline |
|--------|-----------------------|--------------------------------|
| **AUROC** (Novelty) | **0.81** | ~0.50 (random/uncalibrated) |
| **Precision** (Knowns) | **100%** (via Jaccard filter) | 57.9% |
| **Latency per Query** | **~604.9 ms** | ~0.31 ms |
| **Novelty Detection Rate**| **36.8%** true structural novelties | N/A |

*(Note: While the K-mer baseline boasts lower theoretical latency at 0.31ms, its AUROC for open-set detection completely collapses compared to the CNN Embedder, making it unsuitable for real-world environmental sequencing where novel taxa are guaranteed).*

## Hierarchical Elimination Profile
The Search by Elimination ideology successfully prevented forced hallucinations. Out of 500 queries, the pipeline halted classification at the following ranks:
- **Species (Deterministic Match)**: 263
- **Phylum (Novel)**: 41
- **Class (Novel)**: 31
- **Order (Novel)**: 34
- **Family (Novel)**: 40
- **Genus (Known/Novel)**: 91

## FAISS Scalability Benchmark (10,000 Embeddings, 128-D)
To ensure the system could scale to multi-million sequence reference databases (e.g., SILVA), we benchmarked the FAISS backends. `IndexHNSWFlat` was identified as the optimal index, achieving massive throughput with negligible memory footprint.
- **IndexFlatIP**: 21,073 queries/sec (4.90 MB overhead)
- **IndexHNSWFlat**: **247,495 queries/sec (10.67 MB overhead)**
- **IndexIVFFlat**: 178,876 queries/sec (8.83 MB overhead)

## Visualizations
The resulting separation between known sequence embeddings and structurally novel sequences can be visualized via our generated 2D projections. 
*(These plots are dynamically generated via `python paper_plots.py` during your pipeline run).*
- **[View UMAP Embedding Plot](file:///e:/HTRP/paper_plot_scatter.png)**: This plot highlights how the Supervised Contrastive CNN successfully clusters known taxonomic groups, leaving the red "Novel (Test)" sequences isolated in the latent space where the Z-score threshold naturally flags them.
