import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_sequence(seq, k=6, max_len=512):
    seq = seq.upper()
    valid_chars = set('ACGT')
    seq = ''.join([c if c in valid_chars else 'N' for c in seq])
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = []
    for kmer in kmers:
        if 'N' in kmer:
            encoded.append(0)
            continue
        idx = 0
        for i, nuc in enumerate(reversed(kmer)):
            idx += vocab[nuc] * (4 ** i)
        encoded.append(idx + 1)
    
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = encoded + [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

def run_phase9E():
    marine_fa = "case_study_marine.fasta"
    if not os.path.exists(marine_fa):
        print("case_study_marine.fasta not found.")
        return
        
    records = list(SeqIO.parse(marine_fa, "fasta"))
    print(f"Loaded {len(records)} marine sequences.")
    
    import sys
    sys.path.append(os.getcwd())
    from app import CNNEmbedder as SupConModel
    
    results = []
    
    # We will score the marine sequences against the pre-computed indexes
    # We'll use the 'family' holdout rank, seed 42 as the representative configuration 
    # (or we could ensemble across all 20, but the prompt says 'apply frozen parameters')
    # Let's ensemble across all 20 configurations.
    
    ranks = ['class', 'order', 'family', 'genus']
    seeds = [42, 43, 44, 45, 46]
    
    # Pre-encode all sequences so we don't do it 20 times
    print("Pre-encoding marine sequences...")
    X_enc = []
    for r in tqdm(records):
        X_enc.append(encode_sequence(str(r.seq)))
    X_enc = torch.cat(X_enc, dim=0)
    
    for holdout_rank in ranks:
        # Load tax map for this rank
        tax_map = {}
        tax_file = f"assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
        
        rank_idx_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
        target_idx = rank_idx_map.get(holdout_rank.lower(), 5)
        
        with open(tax_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > target_idx:
                    acc = parts[0].split('.')[0]
                    tax_map[acc] = parts[target_idx]
                    
        for seed in seeds:
            print(f"Evaluating Marine eDNA on {holdout_rank} | seed {seed}")
            
            model_path = f"assets/HTRP_models/embedding/16S/{holdout_rank}/seed_{seed}/embedding.pt"
            index_path = f"assets/HTRP_indexes/16S/{holdout_rank}/seed_{seed}/index.faiss"
            ids_path = f"assets/HTRP_indexes/16S/{holdout_rank}/seed_{seed}/ids.npy"
            stats_path = f"assets/HTRP_indexes/16S/{holdout_rank}/seed_{seed}/meta.json"
            
            if not os.path.exists(model_path): continue
            
            model = SupConModel()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            index = faiss.read_index(index_path)
            ids = np.load(ids_path, allow_pickle=True)
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                
            batch_size = 128
            embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(X_enc), batch_size):
                    batch = X_enc[i:i+batch_size]
                    emb = model(batch).cpu().numpy()
                    embeddings.append(emb)
                    
            embeddings = np.vstack(embeddings)
            faiss.normalize_L2(embeddings)
            
            # Predict
            D, I = index.search(embeddings, 10)
            
            for i in range(len(embeddings)):
                hits = ids[I[i]].tolist()
                dists = D[i].tolist()
                
                translated_hits = [tax_map.get(hit.split('.')[0], hit.split('.')[0]) for hit in hits]
                label, freq = Counter(translated_hits).most_common(1)[0]
                
                conf = freq / 10.0
                mean_dist = float(dists[0]) # Inner product
                mean_dist = 1.0 - mean_dist # Convert to cosine distance
                z = (mean_dist - stats["mean"]) / stats["std"]
                
                # NOVEL if Z >= 1.0 or Conf < 0.5
                is_novel = (z >= 1.0) or (conf < 0.5)
                
                results.append({
                    'seq_id': records[i].id,
                    'rank': holdout_rank,
                    'seed': seed,
                    'predicted_taxon': label,
                    'z_score': z,
                    'confidence': conf,
                    'is_novel': is_novel
                })

    df = pd.DataFrame(results)
    df.to_csv("phase9_marine_case_study.csv", index=False)
    
    # Calculate aggregate summary
    print("\\n--- Marine eDNA Summary ---")
    agg = df.groupby('seq_id').agg({
        'is_novel': 'mean',
        'z_score': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    agg['final_prediction'] = agg['is_novel'].apply(lambda x: 'NOVEL' if x >= 0.5 else 'KNOWN')
    
    print(f"Total Sequences: {len(agg)}")
    print(f"Predicted NOVEL: {len(agg[agg['final_prediction'] == 'NOVEL'])}")
    print(f"Predicted KNOWN: {len(agg[agg['final_prediction'] == 'KNOWN'])}")
    
    print("Done!")

if __name__ == "__main__":
    run_phase9E()
