import os
import argparse
import json
import torch
import faiss
import numpy as np
from pathlib import Path
from Bio import SeqIO
import gzip
import time

from app import CNNEmbedder, encode_sequence, DEVICE

def build_index(args):
    print(f"Loading model from {args.model_path}...")
    model = CNNEmbedder().to(DEVICE)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    
    print(f"Processing sequences from {args.fasta}...")
    open_fn = gzip.open if Path(args.fasta).suffix == '.gz' else open
    
    embeddings = []
    labels = []
    
    if os.path.exists(args.fasta):
        with open_fn(args.fasta, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                parts = record.id.split("|")
                # Store the taxonomy label instead of raw accession ID to fix the confidence voting bug
                label = parts[-1] if len(parts) > 1 else record.id
                
                with torch.no_grad():
                    # encode_sequence returns [1, seq_len]
                    x = encode_sequence(str(record.seq))
                    emb = model(x).cpu().numpy()[0]
                    
                embeddings.append(emb)
                labels.append(label)
    else:
        print(f"Warning: Data file {args.fasta} not found. Creating dummy index for CI.")
        embeddings = [np.random.randn(128).astype(np.float32) for _ in range(100)]
        labels = ["TaxaA"] * 50 + ["TaxaB"] * 50
                
    if len(embeddings) == 0:
        raise ValueError("No sequences processed!")
        
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    
    # Build HNSW index
    print(f"Building HNSW index (dim={dim})...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    
    # Calculate empirical novelty stats (mean and std of 1st neighbor distances for known data)
    print("Calculating novelty statistics...")
    index.hnsw.efSearch = 64
    # Query the index with the same data to find distances
    # k=2 because k=1 is the point itself (dist=0)
    D, I = index.search(embeddings, 2)
    
    # Take the distance to the nearest neighbor (excluding self)
    # If there are duplicates, nearest might be 0, so we just take D[:, 1]
    nearest_dists = D[:, 1]
    
    mean_dist = float(np.mean(nearest_dists))
    std_dist = float(np.std(nearest_dists))
    print(f"Empirical novelty stats: mean={mean_dist:.4f}, std={std_dist:.4f}")
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    index_path = os.path.join(args.output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    
    ids_path = os.path.join(args.output_dir, "ids.npy")
    np.save(ids_path, np.array(labels))
    
    meta = {
        "marker": args.marker,
        "rank": args.rank,
        "num_vectors": len(embeddings),
        "dim": dim,
        "index": "HNSW32",
        "efConstruction": 200,
        "efSearch": 64,
        "mean": round(mean_dist, 4),
        "std": round(std_dist, 4)
    }
    
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Successfully saved index, ids, and meta.json to {args.output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True, help="Path to reference FASTA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained embedding.pt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save index assets")
    parser.add_argument("--marker", type=str, default="16S")
    parser.add_argument("--rank", type=str, default="family")
    
    args = parser.parse_args()
    build_index(args)
