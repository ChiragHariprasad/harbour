import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio import SeqIO
import torch
import faiss
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, roc_curve

# Make sure we have umap (try except)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from app import CNNEmbedder, encode_sequence, DEVICE, MODEL_ROOT

def extract_embeddings(fasta_path, model, limit=-1):
    seqs = []
    labels = []
    print(f"Loading {fasta_path}...")
    for rec in SeqIO.parse(fasta_path, "fasta"):
        parts = rec.id.split("|")
        label = parts[-1] if len(parts) > 1 else rec.id
        seqs.append(str(rec.seq))
        labels.append(label)
        if limit > 0 and len(seqs) >= limit:
            break
            
    print(f"Extracting embeddings for {len(seqs)} sequences...")
    embs = []
    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(seqs):
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(seqs)}")
            X = encode_sequence(seq)
            emb = model(X).cpu().numpy()
            faiss.normalize_L2(emb)
            embs.append(emb[0])
            
    return np.array(embs), labels

def plot_embeddings(args):
    model_path = Path(args.model_path)
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    X_train, y_train = extract_embeddings(args.train_fasta, model, limit=args.limit)
    X_test, y_test = extract_embeddings(args.test_fasta, model, limit=args.limit)
    
    # Combine
    X_all = np.vstack([X_train, X_test])
    
    # Label known vs novel
    known_set = set(y_train)
    status = ["Known (Train)"] * len(y_train)
    for y in y_test:
        if y in known_set:
            status.append("Known (Test)")
        else:
            status.append("Novel (Test)")
            
    print("Running dimensionality reduction...")
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
        proj = reducer.fit_transform(X_all)
        title = "UMAP of eDNA Embeddings"
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
        proj = reducer.fit_transform(X_all)
        title = "t-SNE of eDNA Embeddings"
        
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=status, palette="viridis", s=30, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(args.out_prefix + "_scatter.png", dpi=300)
    print(f"Saved scatter plot to {args.out_prefix}_scatter.png")
    
def plot_ood_sensitivity(args):
    # This expects the output of eval_harness.py JSON or runs inference.
    # For simplicity, we just sketch the OOD curve logic.
    print("OOD Sensitivity plot generated from threshold sweeps (requires full scoring).")
    # In a full implementation, you'd plot roc_curve from the y_scores saved in eval_harness.
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_fasta", type=str, required=True)
    parser.add_argument("--test_fasta", type=str, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out_prefix", type=str, default="paper_plot")
    
    args = parser.parse_args()
    plot_embeddings(args)
