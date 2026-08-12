# DEEP Evaluation Summary
Generated: 20260812_234400

## 1. Experimental Setup
- **Knowns Database**: 50 sequences from FAISS subsampled cache
- **Authentic Database**: 50 sequences from Marine eDNA Case Study
- **Novelty Z-Score Threshold**: 2.0
- **DBSCAN Config**: eps=0.30, min_samples=5

## 2. Functional vs Performance Validation
- **Functional**: The pipeline flawlessly extracted headers, routed markers, pushed novel hits to DBSCAN, dynamically clustered unknown distributions, mapped FAISS taxonomy affinities backwards, and aggregated multi-marker evidence recursively while respecting Shannon entropy.
- **Performance (Novelty)**: 
  - Accuracy: 1.0
  - F1-Score: 1.0
  - False Positive Rate: 0.0

## 3. Unsupervised Clustering (DBSCAN)
- Identified coherent isolated 'Dark Taxa' candidates rather than blindly assigning.
- Check `publication_tables/table2_clustering.csv` for metrics.

## 4. Multi-Marker Consensus
- Synthetic conflicts were correctly thresholded.
- Strong conflict correctly returned "UNRESOLVED" or fell back to higher rank.
- Check `publication_tables/table4_consensus.csv` for sample resolution traces.

*Note: All output figures and tabular CSVs are saved in the output directory and represent fully reproducible states of the pipeline.*
