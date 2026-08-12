import os
import joblib
import numpy as np
from pathlib import Path
from Bio import SeqIO
import gzip
import time
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class ClassicalBaseline:
    def __init__(self, k=4):
        self.k = k
        self.vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(k, k),
            lowercase=False
        )
        self.model = MultinomialNB()
        self.is_trained = False
        self.training_time = 0
        self.vectorization_time = 0
        
    def _extract_labels(self, records):
        labels = []
        for rec in records:
            parts = rec.id.split("|")
            labels.append(parts[-1] if len(parts) > 1 else rec.id)
        return labels

    def train(self, fasta_file: str):
        print(f"Training Classical Baseline (MultinomialNB, k={self.k}) on {fasta_file}...")
        open_fn = gzip.open if fasta_file.endswith('.gz') else open
        
        sequences = []
        labels = []
        with open_fn(fasta_file, "rt") as f:
            for rec in SeqIO.parse(f, "fasta"):
                sequences.append(str(rec.seq).upper())
                parts = rec.id.split("|")
                labels.append(parts[-1] if len(parts) > 1 else rec.id)
                
        if not sequences:
            print("No data found!")
            return
            
        t0 = time.time()
        X = self.vectorizer.fit_transform(sequences)
        self.vectorization_time = time.time() - t0
        print(f"Vectorized {len(sequences)} sequences into {X.shape[1]} features in {self.vectorization_time:.2f}s")
        
        t1 = time.time()
        self.model.fit(X, labels)
        self.training_time = time.time() - t1
        print(f"Trained MultinomialNB in {self.training_time:.2f}s")
        
        self.is_trained = True
        
    def evaluate(self, fasta_file: str):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        print(f"Evaluating Baseline on {fasta_file}...")
        open_fn = gzip.open if fasta_file.endswith('.gz') else open
        
        sequences = []
        true_labels = []
        with open_fn(fasta_file, "rt") as f:
            for rec in SeqIO.parse(f, "fasta"):
                sequences.append(str(rec.seq).upper())
                parts = rec.id.split("|")
                true_labels.append(parts[-1] if len(parts) > 1 else rec.id)
                
        t0 = time.time()
        X = self.vectorizer.transform(sequences)
        preds = self.model.predict(X)
        eval_time = time.time() - t0
        
        time_per_query_ms = (eval_time / len(sequences)) * 1000
        print(f"Eval complete. Time per query: {time_per_query_ms:.2f}ms")
        
        report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
        return {
            "accuracy": report["accuracy"],
            "macro_avg_f1": report["macro avg"]["f1-score"],
            "weighted_avg_f1": report["weighted avg"]["f1-score"],
            "time_per_query_ms": time_per_query_ms
        }
        
    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "k": self.k
        }, output_path)
        
    def load(self, model_path: str):
        data = joblib.load(model_path)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.k = data["k"]
        self.is_trained = True

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, default="baseline_metrics.json")
    args = parser.parse_args()
    
    baseline = ClassicalBaseline(k=4)
    baseline.train(args.train)
    results = baseline.evaluate(args.test)
    print("Results:", results)
    baseline.save(args.out)
    
    # Save training and evaluation metrics
    metrics = {
        "vectorization_time_s": baseline.vectorization_time,
        "training_time_s": baseline.training_time,
        "total_train_time_s": baseline.vectorization_time + baseline.training_time,
        "evaluation": results
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
