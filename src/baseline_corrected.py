import os
import joblib
import numpy as np
from Bio import SeqIO
import gzip
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

class ClassicalBaselineCorrected:
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
        
    def load_taxonomy(self, tax_file: str, target_rank: str):
        rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
        rank_idx = rank_map.get(target_rank.lower(), 5)
        tax_map = {}
        with open(tax_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > rank_idx:
                    tax_map[parts[0]] = parts[rank_idx]
        return tax_map

    def train(self, fasta_file: str, tax_file: str, target_rank: str):
        print(f"Training Classical Baseline (MultinomialNB, k={self.k}) on {fasta_file} for rank {target_rank}...")
        tax_map = self.load_taxonomy(tax_file, target_rank)
        open_fn = gzip.open if fasta_file.endswith('.gz') else open
        
        sequences = []
        labels = []
        with open_fn(fasta_file, "rt") as f:
            for rec in SeqIO.parse(f, "fasta"):
                seq_str = str(rec.seq).upper()
                original_id = rec.id.split("|")[0]
                if original_id in tax_map:
                    sequences.append(seq_str)
                    labels.append(tax_map[original_id])
                
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
        print(f"Trained MultinomialNB in {self.training_time:.2f}s. Unique classes: {len(self.model.classes_)}")
        
        self.is_trained = True
        
    def predict_with_novelty(self, sequences, novelty_threshold=0.5):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
            
        X = self.vectorizer.transform(sequences)
        probs = self.model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        best_class_indices = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(sequences)):
            novelty_score = 1.0 - max_probs[i]
            predicted_novel = 1 if novelty_score >= novelty_threshold else 0
            
            if predicted_novel == 1:
                predicted_taxon = "NOVEL"
            else:
                predicted_taxon = self.model.classes_[best_class_indices[i]]
                
            results.append({
                "predicted_taxon": predicted_taxon,
                "predicted_novel": predicted_novel,
                "novelty_score": float(novelty_score)
            })
            
        return results

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
    parser.add_argument("--tax", type=str, required=True)
    parser.add_argument("--target_rank", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    baseline = ClassicalBaselineCorrected(k=4)
    baseline.train(args.train, args.tax, args.target_rank)
    baseline.save(args.out)
