import os
import json
import numpy as np
import pandas as pd
from Bio import SeqIO
from src.baseline_corrected import ClassicalBaselineCorrected

base_dir = "E:/HTRP"
tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"

def load_taxonomy_map(target_rank: str, source_rank="family"):
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(target_rank.lower(), 5)
    source_idx = rank_map.get(source_rank.lower(), 5)
    
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
                # Also map source_rank label -> target_rank label for SupCon translation
                # E.g., Atyidae -> Malacostraca
                tax_map[parts[source_idx]] = parts[target_idx]
    return tax_map

def run_pilot():
    print("Running Pilot Validation for Phase 4B")
    
    configs = [("class", 42), ("class", 43), ("family", 42), ("family", 43)]
    results = {}
    
    for rank, seed in configs:
        print(f"\n--- Pilot: {rank} seed {seed} ---")
        train_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
        eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
        
        baseline = ClassicalBaselineCorrected(k=4)
        baseline.train(train_fa, tax_file, rank)
        
        print(f"Sample classes for {rank}:", baseline.model.classes_[:3])
        
        # Load eval sequences
        eval_records = list(SeqIO.parse(eval_fa, "fasta"))
        eval_seqs = [str(r.seq).upper() for r in eval_records]
        
        preds_raw = baseline.predict_with_novelty(eval_seqs, novelty_threshold=0.5)
        preds = [p["predicted_taxon"] for p in preds_raw]
        
        results[f"{rank}_{seed}"] = preds
        
    print("\n--- Pilot Assertions ---")
    
    # Assert classes are different
    preds_class_42 = results["class_42"]
    preds_family_42 = results["family_42"]
    
    if len(preds_class_42) == len(preds_family_42):
        is_identical = np.array_equal(preds_class_42, preds_family_42)
        print(f"Are Class 42 and Family 42 predictions identical? {is_identical}")
        assert not is_identical, "Predictions are still mechanically reused!"
    
    print("PILOT VALIDATION PASSED!")

if __name__ == "__main__":
    run_pilot()
