import os
import argparse
from pathlib import Path
from Bio import SeqIO
import random
import gzip
import json
import hashlib
from collections import defaultdict

def get_hash(seq_str):
    return hashlib.sha256(seq_str.encode('utf-8')).hexdigest()

def split_fasta(input_file: str, train_file: str, val_file: str, test_file: str, tax_file: str, manifest_file: str, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42, rank_idx: int = 5, holdout_rank: int = 5, holdout_ratio: float = 0.1):
    random.seed(seed)
    
    in_path = Path(input_file)
    if not in_path.exists():
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Reading taxonomy from {tax_file}...")
    tax_map = {}
    holdout_tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > rank_idx:
                tax_map[parts[0]] = parts[rank_idx]
            if holdout_rank != -1 and len(parts) > holdout_rank:
                holdout_tax_map[parts[0]] = parts[holdout_rank]

    # Decide which taxonomic groups to hold out completely for test set
    heldout_groups = set()
    if holdout_rank != -1:
        unique_groups = list(set(holdout_tax_map.values()))
        num_holdout = max(1, int(len(unique_groups) * holdout_ratio))
        heldout_groups = set(random.sample(unique_groups, num_holdout))
        print(f"Holding out {num_holdout} entire groups at rank {holdout_rank} for true open-set evaluation.")

    print(f"Reading {input_file} and de-duplicating sequences...")
    open_fn = gzip.open if in_path.suffix == '.gz' else open
    
    # Group records by holdout_tax
    groups = defaultdict(list)
    seen_seqs = set()
    
    with open_fn(input_file, "rt") as f:
        for rec in SeqIO.parse(f, "fasta"):
            seq_str = str(rec.seq).upper()
            if seq_str in seen_seqs:
                continue
            seen_seqs.add(seq_str)
            
            if rec.id in tax_map:
                label = tax_map[rec.id]
                rec_id = rec.id
                rec.id = f"{rec.id}|{label}"
                rec.description = ""
                
                group_val = holdout_tax_map.get(rec_id, "Unknown") if holdout_rank != -1 else label
                groups[group_val].append(rec)
                
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    val_groups_count = int(len(group_keys) * val_ratio)
    test_groups_count = int(len(group_keys) * test_ratio)
    
    val_keys = group_keys[:val_groups_count]
    test_keys = group_keys[val_groups_count:val_groups_count+test_groups_count]
    train_keys = group_keys[val_groups_count+test_groups_count:]
    
    train_records = [rec for k in train_keys for rec in groups[k]]
    val_records = [rec for k in val_keys for rec in groups[k]]
    test_records = [rec for k in test_keys for rec in groups[k]]
    
    random.shuffle(train_records)
    random.shuffle(test_records)
    
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    if val_file: os.makedirs(os.path.dirname(val_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    print(f"Writing {len(train_records)} to {train_file}...")
    SeqIO.write(train_records, train_file, "fasta")
    
    if val_file:
        print(f"Writing {len(val_records)} to {val_file}...")
        SeqIO.write(val_records, val_file, "fasta")
        
    print(f"Writing {len(test_records)} to {test_file}...")
    SeqIO.write(test_records, test_file, "fasta")
    
    # Generate manifest
    manifest = {
        "dataset_version": "2.0.0_strict_dedup",
        "source_file": input_file,
        "split_seed": seed,
        "taxonomic_holdout_rank": holdout_rank,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "test_count": len(test_records),
        "held_out_taxa": test_keys,
        "file_hashes": {
            "train": get_hash(open(train_file, "r").read()),
            "val": get_hash(open(val_file, "r").read()) if val_file else None,
            "test": get_hash(open(test_file, "r").read())
        }
    }
    
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to {manifest_file}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input FASTA")
    parser.add_argument("--train", type=str, required=True, help="Output Train FASTA")
    parser.add_argument("--val", type=str, default="", help="Output Val FASTA")
    parser.add_argument("--test", type=str, required=True, help="Output Test FASTA")
    parser.add_argument("--tax", type=str, required=True, help="Taxonomy file mapping")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest JSON output")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio (closed set)")
    parser.add_argument("--rank_idx", type=int, default=5, help="Taxonomic rank index")
    parser.add_argument("--holdout_rank", type=int, default=5, help="Taxonomic rank to hold out entire groups")
    parser.add_argument("--holdout_ratio", type=float, default=0.1, help="Ratio of unique groups to hold out completely")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()
    split_fasta(args.input, args.train, args.val, args.test, args.tax, args.manifest, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed, rank_idx=args.rank_idx, holdout_rank=args.holdout_rank, holdout_ratio=args.holdout_ratio)
