import math
from collections import Counter
import numpy as np

def calculate_entropy(seq: str, k=3) -> float:
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1) if "N" not in seq[i:i+k]]
    if not kmers:
        return 0.0
        
    counts = Counter(kmers)
    total = len(kmers)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy

def sequence_qc(seq: str):
    """
    Cheap deterministic QC gate.
    Returns (is_corrupted, details)
    """
    seq = seq.upper()
    length = len(seq)
    
    if length < 50:
        return True, {"reason": "Too short", "length": length}
        
    n_count = seq.count("N")
    n_fraction = n_count / length
    if n_fraction > 0.05:
        return True, {"reason": "Too many ambiguous bases", "n_fraction": round(n_fraction, 3)}
        
    entropy = calculate_entropy(seq)
    if entropy < 1.5:  # Arbitrary threshold for repetitive sequences
        return True, {"reason": "Low k-mer entropy", "entropy": round(entropy, 3)}
        
    # Placeholder for learned corruption classifier
    # A real model would output p(corrupted) here
    corrupted_probability = 0.02
    
    if corrupted_probability > 0.5:
        return True, {"reason": "Learned corruption model flagged", "probability": corrupted_probability}
        
    return False, {"corrupted_probability": corrupted_probability, "flags": []}
