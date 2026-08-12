import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def kmer_tokenizer(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def run_kmer_ml_baseline(args):
    print("Loading training data...")
    train_seqs = []
    train_labels = []
    for rec in SeqIO.parse(args.train_fasta, "fasta"):
        parts = rec.id.split("|")
        label = parts[-1] if len(parts) > 1 else rec.id
        train_seqs.append(str(rec.seq))
        train_labels.append(label)
        
        if args.train_limit > 0 and len(train_seqs) >= args.train_limit:
            break
            
    print(f"Loaded {len(train_seqs)} training sequences. Training CountVectorizer...")
    vectorizer = CountVectorizer(analyzer=lambda x: kmer_tokenizer(x, k=4), max_features=1000)
    X_train = vectorizer.fit_transform(train_seqs)
    
    print("Training Random Forest Classifier...")
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    clf.fit(X_train, train_labels)
    train_time = time.time() - t0
    print(f"Trained in {train_time:.2f}s")
    
    known_taxa = set(train_labels)
    
    print("Loading test data...")
    test_seqs = []
    y_true = []
    
    for rec in SeqIO.parse(args.test_fasta, "fasta"):
        parts = rec.id.split("|")
        label = parts[-1] if len(parts) > 1 else rec.id
        test_seqs.append(str(rec.seq))
        y_true.append(0 if label in known_taxa else 1) # 1 if novel
        
        if args.test_limit > 0 and len(test_seqs) >= args.test_limit:
            break
            
    print(f"Evaluating {len(test_seqs)} queries...")
    
    X_test = vectorizer.transform(test_seqs)
    
    # We will use the max probability to define novelty. Lower max prob = more likely novel.
    t0 = time.time()
    probs = clf.predict_proba(X_test)
    y_pred_probs = np.max(probs, axis=1) 
    y_pred_classes = clf.classes_[np.argmax(probs, axis=1)]
    inference_time = time.time() - t0
    
    # Anomaly score is 1.0 - max_prob
    y_score = 1.0 - y_pred_probs
    y_pred = (y_score > 0.5).astype(int) # Predict novel if max prob < 0.5
    
    avg_latency = inference_time / len(test_seqs) * 1000
    
    y_true = np.array(y_true)
    
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
    else:
        auroc = 0.0
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n===== KMER+ML BASELINE METRICS =====")
    print(f"Accuracy:        {acc:.4f}")
    print(f"F1-Score:        {f1:.4f}")
    print(f"AUROC:           {auroc:.4f}")
    print(f"Avg Latency:     {avg_latency:.2f} ms")
    
    out = {
        "baseline": "Kmer_RandomForest",
        "train_time_s": train_time,
        "total_queries": len(test_seqs),
        "average_latency_ms": avg_latency,
        "metrics": {
            "accuracy": acc,
            "f1": f1,
            "auroc": auroc
        }
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fasta", type=str, required=True)
    parser.add_argument("--test_fasta", type=str, required=True)
    parser.add_argument("--train_limit", type=int, default=-1)
    parser.add_argument("--test_limit", type=int, default=-1)
    parser.add_argument("--out", type=str, default="baseline_kmer.json")
    
    args = parser.parse_args()
    run_kmer_ml_baseline(args)
