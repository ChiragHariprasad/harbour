import numpy as np
import joblib
from pathlib import Path

VALID_MARKERS = {"16S", "18S", "CO1", "FITS", "PITS", "trnL", "Vert12S"}

class MarkerDetector:
    def __init__(self, model_path: Path = None):
        self.k = 4
        self.vocab_size = 4**self.k
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.model = None
        self.classes_ = []
        
        if model_path and model_path.exists():
            data = joblib.load(model_path)
            self.model = data["model"]
            self.classes_ = data["classes"]
            
    def _encode_kmer(self, kmer: str) -> int:
        idx = 0
        for char in kmer:
            if char not in self.mapping:
                return -1
            idx = idx * 4 + self.mapping[char]
        return idx
        
    def extract_features(self, seq: str) -> np.ndarray:
        seq = seq.upper()
        counts = np.zeros(self.vocab_size, dtype=np.float32)
        total = 0
        for i in range(len(seq) - self.k + 1):
            idx = self._encode_kmer(seq[i:i+self.k])
            if idx != -1:
                counts[idx] += 1
                total += 1
        
        if total > 0:
            counts /= total
            
        return counts.reshape(1, -1)
        
    def detect(self, seq: str):
        if self.model is None:
             return {"label": "16S", "confidence": 0.51}
             
        features = self.extract_features(seq)
        probas = self.model.predict_proba(features)[0]
        max_idx = np.argmax(probas)
        
        return {
            "label": self.classes_[max_idx],
            "confidence": round(float(probas[max_idx]), 4)
        }
