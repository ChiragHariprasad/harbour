import numpy as np
import faiss
from typing import List, Dict, Any

class DarkTaxaClusterer:
    def __init__(self, eps: float = 0.30, min_samples: int = 5, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def _calculate_medoid(self, embeddings: np.ndarray) -> int:
        """
        Calculate the index of the medoid (the point with the minimum average distance to all other points).
        """
        n = len(embeddings)
        if n == 0:
            return -1
        
        # Calculate pairwise L2 distances
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                
        avg_dists = np.mean(dist_matrix, axis=1)
        return int(np.argmin(avg_dists))

    def _get_taxonomic_affinity(self, medoid_emb: np.ndarray, index, ids) -> Dict[str, Any]:
        """
        Search FAISS for the closest known reference neighbors to establish taxonomic affinity.
        """
        # Search FAISS
        D, I = index.search(medoid_emb.reshape(1, -1), 10)
        hits = ids[I[0]].tolist()
        
        if not hits:
            return None
            
        from collections import Counter
        label, count = Counter(hits).most_common(1)[0]
        support = count / 10.0
        
        return {
            "rank": "genus", # Assuming the index is at genus or we just return the label
            "label": label,
            "support": support
        }

    def cluster(self, candidates: List[Dict[str, Any]], marker: str, index, ids) -> Dict[str, Any]:
        """
        Cluster a pool of novel candidates.
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return {"error": "scikit-learn is required for clustering"}
            
        if not candidates:
            return {"candidate_count": 0, "clusters": []}

        # Extract embeddings
        embeddings = []
        valid_candidates = []
        for c in candidates:
            emb = c.get("_embedding")
            if emb is not None:
                embeddings.append(emb)
                valid_candidates.append(c)
                
        if not embeddings:
            return {"candidate_count": len(candidates), "clusters": []}
            
        embeddings_np = np.vstack(embeddings)
        
        # DBSCAN expects normalized vectors for cosine-like distance thresholding (L2 = sqrt(2*(1-cos)))
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        labels = clustering.fit_predict(embeddings_np)
        
        clusters_meta = []
        noise_count = 0
        
        for candidate, label in zip(valid_candidates, labels):
            candidate["novelty_cluster"] = int(label)
            if label == -1:
                noise_count += 1
                
        unique_labels = set(labels)
        unique_labels.discard(-1) # ignore noise
        
        for cluster_id in unique_labels:
            indices = np.where(labels == cluster_id)[0]
            cluster_embs = embeddings_np[indices]
            
            medoid_idx = self._calculate_medoid(cluster_embs)
            medoid_candidate = valid_candidates[indices[medoid_idx]]
            
            # Compute cluster stats
            z_scores = [valid_candidates[i].get("z_score", 0.0) for i in indices]
            mean_z_score = float(np.mean(z_scores))
            
            # Tax affinity for medoid
            affinity = self._get_taxonomic_affinity(cluster_embs[medoid_idx], index, ids)
            
            clusters_meta.append({
                "cluster_id": int(cluster_id),
                "size": int(len(indices)),
                "marker": marker,
                "representative_sequence": medoid_candidate.get("query", "unknown"),
                "mean_z_score": round(mean_z_score, 3),
                "status": "CANDIDATE_NOVEL_GROUP",
                "taxonomic_affinity": affinity
            })
            
        return {
            "enabled": True,
            "mode": "fixed",
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "candidate_count": len(valid_candidates),
            "cluster_count": len(clusters_meta),
            "noise_count": noise_count,
            "clusters": clusters_meta
        }
