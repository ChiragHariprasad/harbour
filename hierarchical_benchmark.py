import os
import json
import time
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from Bio import SeqIO
import random

from hierarchical_infer import HierarchicalPipeline

def run_benchmark(args):
    print("Loading test data...")
    test_seqs = []
    test_labels = []
    
    # We will test against the GENUS test set to have the most granular labels
    test_fasta = Path(f"assets/HTRP_cache/{args.marker}/genus/test.fasta")
    
    # We load all test sequences
    for rec in SeqIO.parse(test_fasta, "fasta"):
        parts = rec.id.split("|")
        # Genus is the last part
        test_labels.append(parts[-1] if len(parts) > 1 else "Unknown")
        test_seqs.append(str(rec.seq))
        
    # We sample if needed
    if len(test_seqs) > args.samples:
        indices = random.sample(range(len(test_seqs)), args.samples)
        test_seqs = [test_seqs[i] for i in indices]
        test_labels = [test_labels[i] for i in indices]
        
    print(f"Sampled {len(test_seqs)} queries from {test_fasta}")
    
    pipeline = HierarchicalPipeline(marker=args.marker)
    
    print(f"Evaluating Hierarchical Pipeline...")
    
    preds = []
    times = []
    paths = []
    levels = []
    
    known_count = 0
    novel_count = 0
    
    for i, seq in enumerate(test_seqs):
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(test_seqs)}")
            
        t0 = time.time()
        res = pipeline.predict(seq)
        t_elapsed = time.time() - t0
        
        pred = res["prediction"]
        path = res["path"]
        level = res["level"]
        
        preds.append(pred)
        times.append(t_elapsed)
        paths.append(path)
        levels.append(level)
        
        if "KNOWN" in pred:
            known_count += 1
        else:
            novel_count += 1
            
    print("\n===== HIERARCHICAL CASCADE RESULTS =====")
    print(f"Total Queries: {len(test_seqs)}")
    print(f"Classified as KNOWN: {known_count} ({known_count/len(test_seqs)*100:.2f}%)")
    print(f"Classified as NOVEL: {novel_count} ({novel_count/len(test_seqs)*100:.2f}%)")
    
    avg_time = sum(times) / len(times) * 1000
    print(f"Average Latency: {avg_time:.2f} ms per query")
    
    # Count paths
    det_path = sum(1 for p in paths if p == "deterministic")
    ml_path = sum(1 for p in paths if p == "ml_cascade")
    print(f"Deterministic hits: {det_path}")
    print(f"ML Cascade hits: {ml_path}")
    
    # Levels of elimination
    from collections import Counter
    level_counts = Counter(levels)
    print("\nElimination Levels:")
    for lvl, cnt in level_counts.items():
        print(f" - {lvl}: {cnt}")
        
    out = {
        "marker": args.marker,
        "total_queries": len(test_seqs),
        "known_percentage": known_count / len(test_seqs) * 100,
        "novel_percentage": novel_count / len(test_seqs) * 100,
        "latency_ms": avg_time,
        "elimination_levels": dict(level_counts)
    }
    
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=str, default="16S")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out", type=str, default="benchmark_hierarchical.json")
    
    args = parser.parse_args()
    run_benchmark(args)
