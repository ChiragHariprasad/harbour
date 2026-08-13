import os
import glob
import json
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clean_label(label):
    if not isinstance(label, str):
        return ""
    label = label.strip()
    if label.lower() in ['uncultured', 'unclassified', 'sp.', 'na', 'n/a', 'unknown', '']:
        return ""
    return label

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

def get_taxonomy_dict(rank_dir):
    tax_file = os.path.join(rank_dir, "subsampled_taxonomy.txt")
    tax_dict = {}
    if not os.path.exists(tax_file):
        return tax_dict
    with open(tax_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                acc = parts[0].split('.')[0]
                lin = parts[1:7] if len(parts) >= 7 else parts[1:]
                while len(lin) < 6:
                    lin.append("")
                tax_dict[acc] = lin
    return tax_dict

def run_phase9C():
    # Load Protocol Freeze Manifest
    with open('phase9_freeze_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    RANKS = manifest['frozen_parameters']['taxonomic_rank_ordering']
    SEEDS = manifest['frozen_parameters']['random_seeds']
    K = manifest['frozen_parameters']['nn_purity']['k']
    
    import sys
    sys.path.append(os.getcwd())
    from app import CNNEmbedder as SupConModel
    
    raw_predictions = []
    summaries = []
    
    tax_dir = "assets/HTRP_cache_LEAKED_P2/16S/family"
    tax_dict = get_taxonomy_dict(tax_dir)
    for holdout_rank in ['family', 'genus', 'order', 'class']:
        rank_dir = f"assets/HTRP_cache/16S/{holdout_rank}"
        
        for seed in SEEDS:
            print(f"\\nProcessing Phase 9C: 16S | {holdout_rank} | seed {seed}")
            
            model_path = f"assets/HTRP_models/embedding/16S/{holdout_rank}/seed_{seed}/embedding.pt"
            if not os.path.exists(model_path):
                continue
                
            model = SupConModel()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            test_fasta = f"{rank_dir}/seed_{seed}/test.fasta"
            if not os.path.exists(test_fasta):
                continue
                
            records = list(SeqIO.parse(test_fasta, "fasta"))
            
            embeddings = []
            lineages = []
            seq_ids = []
            
            print(f"Encoding {len(records)} test sequences...")
            with torch.no_grad():
                for r in tqdm(records, desc="Encoding"):
                    seq_id = r.id.split('|')[0]
                    acc = seq_id.split('.')[0]
                    if acc not in tax_dict:
                        continue
                    
                    enc = encode_sequence(str(r.seq))
                    emb = model(enc).cpu().numpy()[0]
                    
                    embeddings.append(emb)
                    lineages.append(tax_dict[acc])
                    seq_ids.append(seq_id)
            
            embeddings = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)
            
            # Build index for self-exclusion search
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            # We search for k+1 to allow self-exclusion
            D, I = index.search(embeddings, K + 1)
            
            class_purity = []
            order_purity = []
            family_purity = []
            genus_purity = []
            
            for i in range(len(embeddings)):
                query_lin = lineages[i]
                q_class = clean_label(query_lin[1])
                q_order = clean_label(query_lin[2])
                q_family = clean_label(query_lin[3])
                q_genus = clean_label(query_lin[4])
                
                # Exclude self
                neighbors = I[i]
                dists = D[i]
                
                # Filter out the query index
                valid_idx = [j for j, idx in enumerate(neighbors) if idx != i]
                # Take exactly K neighbors
                valid_idx = valid_idx[:K]
                
                c_matches = 0
                o_matches = 0
                f_matches = 0
                g_matches = 0
                
                for rank_idx, j in enumerate(valid_idx):
                    n_idx = neighbors[j]
                    n_dist = 1.0 - dists[j] # Convert inner product to cosine distance
                    n_lin = lineages[n_idx]
                    
                    n_class = clean_label(n_lin[1])
                    n_order = clean_label(n_lin[2])
                    n_family = clean_label(n_lin[3])
                    n_genus = clean_label(n_lin[4])
                    
                    # Missing labels don't count as a match or mismatch towards the denominator.
                    # Wait, the prompt says "calculate proportion sharing Class Order Family Genus".
                    # Let's consider standard purity: if either query or neighbor is missing, it's not a match.
                    if q_class and n_class and q_class == n_class: c_matches += 1
                    if q_order and n_order and q_order == n_order: o_matches += 1
                    if q_family and n_family and q_family == n_family: f_matches += 1
                    if q_genus and n_genus and q_genus == n_genus: g_matches += 1
                    
                    raw_predictions.append({
                        'experiment_id': f"{holdout_rank}_{seed}",
                        'rank': holdout_rank,
                        'seed': seed,
                        'sequence_id': seq_ids[i],
                        'neighbor_rank': rank_idx + 1,
                        'neighbor_sequence_id': seq_ids[n_idx],
                        'embedding_distance': n_dist,
                        'neighbor_taxon_class': n_class,
                        'neighbor_taxon_order': n_order,
                        'neighbor_taxon_family': n_family,
                        'neighbor_taxon_genus': n_genus
                    })
                
                class_purity.append(c_matches / K)
                order_purity.append(o_matches / K)
                family_purity.append(f_matches / K)
                genus_purity.append(g_matches / K)
                
            summaries.append({
                'rank': holdout_rank,
                'seed': seed,
                'class_purity_mean': np.mean(class_purity),
                'class_purity_sd': np.std(class_purity),
                'order_purity_mean': np.mean(order_purity),
                'order_purity_sd': np.std(order_purity),
                'family_purity_mean': np.mean(family_purity),
                'family_purity_sd': np.std(family_purity),
                'genus_purity_mean': np.mean(genus_purity),
                'genus_purity_sd': np.std(genus_purity)
            })
            
    df_raw = pd.DataFrame(raw_predictions)
    df_raw.to_csv("phase9_nn_purity_predictions.csv", index=False)
    
    df_summ = pd.DataFrame(summaries)
    df_summ.to_csv("phase9_nn_purity_summary.csv", index=False)
    
    print("Done! Saved phase9_nn_purity_predictions.csv and phase9_nn_purity_summary.csv")

if __name__ == "__main__":
    run_phase9C()
