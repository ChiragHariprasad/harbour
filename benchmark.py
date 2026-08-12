import argparse
import time
import json
from pathlib import Path
from Bio import SeqIO
import numpy as np

from app import run_pipeline, get_model, get_index, MODELS, INDEXES, infer_ml
from src.baseline import ClassicalBaseline

def evaluate_predictions(true_labels, preds):
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

def run_benchmark(test_fasta: str, marker: str, baseline_path: str):
    print(f"--- Running Benchmark for {marker} ---")
    
    # 1. Load Data
    sequences = []
    true_labels = []
    for rec in SeqIO.parse(test_fasta, "fasta"):
        sequences.append(str(rec.seq))
        parts = rec.id.split("|")
        true_labels.append(parts[-1] if len(parts) > 1 else rec.id)
        
    # Limit to 1000 sequences to avoid taking > 1 hour
    if len(sequences) > 1000:
        import random
        random.seed(42)
        combined = list(zip(sequences, true_labels))
        random.shuffle(combined)
        combined = combined[:1000]
        sequences, true_labels = zip(*combined)
        sequences = list(sequences)
        true_labels = list(true_labels)
        
    num_queries = len(sequences)
    print(f"Sampled {num_queries} queries from {test_fasta}")
    
    # 2. Evaluate Classical Baseline
    print("Evaluating Classical Baseline...")
    baseline = ClassicalBaseline()
    try:
        baseline.load(baseline_path)
        baseline_res = baseline.evaluate(test_fasta)
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        baseline_res = {"accuracy": 0, "macro_avg_f1": 0, "time_per_query_ms": 0}
        
    # 3. Evaluate ML Pipeline (CNNEmbedder)
    print("Evaluating HARBOUR CNN Pipeline (ML Path)...")
    pipeline_preds = []
    t0 = time.time()
    
    for seq in sequences:
        # Bypass deterministic matcher to benchmark the ML embedding model directly
        res = infer_ml(seq, marker)
        
        # Get family prediction
        pred_label = "NOVEL"
        if "family" in res:
            pred_label = res["family"]["prediction"]
        
        pipeline_preds.append(pred_label)
        
    pipeline_time = time.time() - t0
    pipeline_time_ms = (pipeline_time / num_queries) * 1000
    
    # Calculate metrics
    pipeline_metrics = evaluate_predictions(true_labels, pipeline_preds)
    
    print(f"\n===== RESULTS FOR {marker} =====")
    print(f"Classical Baseline:")
    print(f"  Accuracy:    {baseline_res['accuracy']:.4f}")
    print(f"  Macro F1:    {baseline_res['macro_avg_f1']:.4f}")
    print(f"  Latency/Q:   {baseline_res['time_per_query_ms']:.2f} ms")
    print(f"")
    print(f"HARBOUR CNN (Current loaded model):")
    print(f"  Accuracy:    {pipeline_metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {pipeline_metrics['macro_f1']:.4f}")
    print(f"  Latency/Q:   {pipeline_time_ms:.2f} ms")
    
    return {
        "marker": marker,
        "baseline": baseline_res,
        "harbour_cnn": {
            "accuracy": pipeline_metrics["accuracy"],
            "macro_f1": pipeline_metrics["macro_f1"],
            "time_per_query_ms": pipeline_time_ms
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--marker", type=str, default="16S")
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--out", type=str, default="benchmark_results.json")
    
    args = parser.parse_args()
    results = run_benchmark(args.test, args.marker, args.baseline_model)
    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
