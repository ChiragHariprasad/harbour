import os
import glob
import json
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Protocol Freeze Manifest
with open('phase9_freeze_manifest.json', 'r') as f:
    manifest = json.load(f)

RANKS = manifest['frozen_parameters']['taxonomic_rank_ordering']
SEEDS = manifest['frozen_parameters']['random_seeds']
MAX_PAIRS = manifest['frozen_parameters']['pair_sampling_algorithm']['max_pairs_per_config']

def clean_label(label):
    if not isinstance(label, str):
        return ""
    label = label.strip()
    if label.lower() in ['uncultured', 'unclassified', 'sp.', 'na', 'n/a', 'unknown', '']:
        return ""
    return label

def compute_dtax(lin1, lin2):
    """
    Computes D_tax (0-6) based on divergence.
    Missing/invalid labels at a rank do NOT cause a mismatch.
    Returns the number of boundaries at which they diverge.
    0 = species match
    6 = phylum mismatch
    """
    for i in range(len(RANKS)):
        l1 = clean_label(lin1[i])
        l2 = clean_label(lin2[i])
        if l1 == "" or l2 == "":
            continue # Don't mismatch on missing data
        if l1 != l2:
            return 6 - i # e.g. Phylum mismatch -> 6-0 = 6
    return 0 # Exact match at all valid levels

def encode_sequence(seq, k=6, max_len=512):
    """K-mer tokenizer matching the standard pipeline."""
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

def run_phase9B():
    np.random.seed(42)
    
    results = []
    pair_samples = []
    
    # Load model class
    import sys
    sys.path.append(os.getcwd())
    try:
        from app import CNNEmbedder as SupConModel
    except ImportError:
        print("Could not import CNNEmbedder from app.py")
        return
        
    tax_dir = "assets/HTRP_cache_LEAKED_P2/16S/family"
    tax_dict = get_taxonomy_dict(tax_dir)
    for holdout_rank in ['family', 'genus', 'order', 'class']:
        rank_dir = f"assets/HTRP_cache/16S/{holdout_rank}"
        
        for seed in SEEDS:
            print(f"\\nProcessing 16S | {holdout_rank} | seed {seed}")
            
            # Load embeddings
            model_path = f"assets/HTRP_models/embedding/16S/{holdout_rank}/seed_{seed}/embedding.pt"
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
                
            model = SupConModel()
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            # Load test sequences
            test_fasta = f"{rank_dir}/seed_{seed}/test.fasta"
            if not os.path.exists(test_fasta):
                continue
                
            records = list(SeqIO.parse(test_fasta, "fasta"))
            # Subset to save compute if too huge, but usually ~5000 is fine
            
            embeddings = []
            lineages = []
            seq_ids = []
            
            print(f"Encoding {len(records)} sequences...")
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
            
            embeddings = np.array(embeddings)
            # L2 normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # We want up to MAX_PAIRS pairs
            n_seqs = len(embeddings)
            
            # Generate random pairs
            # To get enough representation across all D_tax levels, we can sample a large random pool and bucket them
            print("Generating pairs...")
            pool_size = 500000 # Sample half a million random pairs to bucket
            idx1 = np.random.randint(0, n_seqs, pool_size)
            idx2 = np.random.randint(0, n_seqs, pool_size)
            
            # Filter self pairs
            valid_mask = idx1 != idx2
            idx1 = idx1[valid_mask]
            idx2 = idx2[valid_mask]
            
            dist_tax = []
            for i, j in zip(idx1, idx2):
                dist_tax.append(compute_dtax(lineages[i], lineages[j]))
            
            dist_tax = np.array(dist_tax)
            
            # 1. Balanced Sampling
            balanced_idx1 = []
            balanced_idx2 = []
            balanced_dtax = []
            
            # Find unique levels
            levels = np.unique(dist_tax)
            target_per_level = MAX_PAIRS // len(levels)
            
            for lvl in levels:
                lvl_mask = (dist_tax == lvl)
                lvl_idx1 = idx1[lvl_mask]
                lvl_idx2 = idx2[lvl_mask]
                
                n_available = len(lvl_idx1)
                n_sample = min(n_available, target_per_level)
                
                if n_sample > 0:
                    chosen = np.random.choice(n_available, n_sample, replace=False)
                    balanced_idx1.extend(lvl_idx1[chosen])
                    balanced_idx2.extend(lvl_idx2[chosen])
                    balanced_dtax.extend([lvl] * n_sample)
                    
            balanced_idx1 = np.array(balanced_idx1)
            balanced_idx2 = np.array(balanced_idx2)
            balanced_dtax = np.array(balanced_dtax)
            
            # Compute embeddings dists for balanced
            print("Computing balanced embedding distances...")
            emb_dists_balanced = 1.0 - np.sum(embeddings[balanced_idx1] * embeddings[balanced_idx2], axis=1)
            
            spearman_bal, p_bal_s = spearmanr(emb_dists_balanced, balanced_dtax)
            pearson_bal, p_bal_p = pearsonr(emb_dists_balanced, balanced_dtax)
            
            # Bootstrap CIs for Balanced
            n_boot = 1000
            spearman_bal_boots = []
            n_bal = len(balanced_dtax)
            for _ in range(n_boot):
                b_idx = np.random.randint(0, n_bal, n_bal)
                rho, _ = spearmanr(emb_dists_balanced[b_idx], balanced_dtax[b_idx])
                spearman_bal_boots.append(rho)
            ci_bal_lower, ci_bal_upper = np.percentile(spearman_bal_boots, [2.5, 97.5])
            
            # 2. Natural Sampling
            # Take a random subset of natural distribution matching the size of balanced
            n_natural = len(balanced_dtax)
            natural_chosen = np.random.choice(len(idx1), n_natural, replace=False)
            
            natural_idx1 = idx1[natural_chosen]
            natural_idx2 = idx2[natural_chosen]
            natural_dtax = dist_tax[natural_chosen]
            
            print("Computing natural embedding distances...")
            emb_dists_natural = 1.0 - np.sum(embeddings[natural_idx1] * embeddings[natural_idx2], axis=1)
            
            spearman_nat, p_nat_s = spearmanr(emb_dists_natural, natural_dtax)
            pearson_nat, p_nat_p = pearsonr(emb_dists_natural, natural_dtax)
            
            spearman_nat_boots = []
            for _ in range(n_boot):
                b_idx = np.random.randint(0, n_natural, n_natural)
                rho, _ = spearmanr(emb_dists_natural[b_idx], natural_dtax[b_idx])
                spearman_nat_boots.append(rho)
            ci_nat_lower, ci_nat_upper = np.percentile(spearman_nat_boots, [2.5, 97.5])
            
            # Save stats
            results.append({
                'rank': holdout_rank,
                'seed': seed,
                'balanced_spearman': spearman_bal,
                'balanced_pearson': pearson_bal,
                'balanced_n_pairs': n_bal,
                'balanced_ci_lower': ci_bal_lower,
                'balanced_ci_upper': ci_bal_upper,
                'natural_spearman': spearman_nat,
                'natural_pearson': pearson_nat,
                'natural_n_pairs': n_natural,
                'natural_ci_lower': ci_nat_lower,
                'natural_ci_upper': ci_nat_upper
            })
            
            # Save samples
            for i in range(len(balanced_dtax)):
                pair_samples.append({
                    'rank': holdout_rank, 'seed': seed, 'type': 'balanced',
                    'seq1': seq_ids[balanced_idx1[i]], 'seq2': seq_ids[balanced_idx2[i]],
                    'd_tax': balanced_dtax[i], 'd_embed': emb_dists_balanced[i]
                })
            for i in range(len(natural_dtax)):
                pair_samples.append({
                    'rank': holdout_rank, 'seed': seed, 'type': 'natural',
                    'seq1': seq_ids[natural_idx1[i]], 'seq2': seq_ids[natural_idx2[i]],
                    'd_tax': natural_dtax[i], 'd_embed': emb_dists_natural[i]
                })

    df_res = pd.DataFrame(results)
    df_res.to_csv("phase9_embedding_taxonomy_correlations.csv", index=False)
    
    df_samples = pd.DataFrame(pair_samples)
    df_samples.to_csv("phase9_pair_samples.csv", index=False)
    
    print("Done! Saved correlations to phase9_embedding_taxonomy_correlations.csv")

if __name__ == "__main__":
    run_phase9B()
