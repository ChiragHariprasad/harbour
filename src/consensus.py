import math
from collections import defaultdict
from typing import List, Dict, Any

# Hierarchy defined from top to bottom
TAXONOMIC_HIERARCHY = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]

class MultiMarkerHierarchicalConsensus:
    def __init__(self, marker_reliability: Dict[str, Dict[str, float]] = None):
        """
        marker_reliability format:
        {
            "CO1": {"family": 0.97, "genus": 0.93, "species": 0.98},
            ...
        }
        """
        self.marker_reliability = marker_reliability or {}

    def _get_marker_reliability(self, marker: str, rank: str) -> float:
        if marker in self.marker_reliability and rank in self.marker_reliability[marker]:
            return self.marker_reliability[marker][rank]
        return 0.8 # default high reliability if missing/uncalibrated

    def _compute_entropy(self, evidence: Dict[str, float]) -> float:
        total = sum(evidence.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for val in evidence.values():
            p = val / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def resolve(self, sample_id: str, predictions: List[Dict[str, Any]], 
                min_confidence: float = 0.6, max_entropy: float = 1.0) -> Dict[str, Any]:
        """
        predictions: list of results from run_pipeline, must contain 'trace' which has per-rank info.
        """
        # Collect evidence per rank
        evidence_per_rank = defaultdict(lambda: defaultdict(float))
        marker_weights = {}
        
        for pred in predictions:
            marker = pred.get("marker")
            if not marker:
                continue
                
            # Compute Q_m (sequence quality). For now, default to 1.0 if not available
            Q_m = pred.get("quality_score", 1.0)
            
            trace = pred.get("results", {}) # assuming ml_results trace format from app.py
            
            for rank in TAXONOMIC_HIERARCHY:
                if rank in trace:
                    rank_info = trace[rank]
                    raw_prediction = rank_info.get("prediction")
                    
                    if raw_prediction and not raw_prediction.startswith("NOVEL"):
                        R_m = self._get_marker_reliability(marker, rank)
                        # We use the raw confidence here and assume it's somewhat calibrated
                        C_m = rank_info.get("confidence", 0.0)
                        
                        W_m = Q_m * R_m * C_m
                        marker_weights[f"{marker}_{rank}"] = W_m
                        
                        evidence_per_rank[rank][raw_prediction] += W_m
                        
        consensus_taxonomy = {}
        resolved_rank = None
        conflicts = []
        agreements = {}
        entropies = {}
        
        for rank in TAXONOMIC_HIERARCHY:
            if rank not in evidence_per_rank or not evidence_per_rank[rank]:
                break
                
            rank_evidence = evidence_per_rank[rank]
            total_evidence = sum(rank_evidence.values())
            
            if total_evidence == 0:
                break
                
            best_label = max(rank_evidence, key=rank_evidence.get)
            best_evidence = rank_evidence[best_label]
            
            confidence = best_evidence / total_evidence
            entropy = self._compute_entropy(rank_evidence)
            
            agreements[rank] = confidence
            entropies[rank] = entropy
            
            if confidence >= min_confidence and entropy <= max_entropy:
                consensus_taxonomy[rank] = best_label
                resolved_rank = rank
            else:
                conflicts.append({
                    "rank": rank,
                    "evidence": dict(rank_evidence),
                    "entropy": entropy
                })
                break
                
        return {
            "sample_id": sample_id,
            "consensus_taxonomy": consensus_taxonomy,
            "resolved_rank": resolved_rank,
            "agreements": agreements,
            "entropies": entropies,
            "conflicts": conflicts,
            "marker_weights": marker_weights
        }
