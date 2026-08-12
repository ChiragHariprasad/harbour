import os
import argparse
from pathlib import Path
from Bio import SeqIO
import random
import gzip

def split_fasta(input_file: str, train_file: str, test_file: str, tax_file: str, test_ratio: float = 0.2, seed: int = 42, rank_idx: int = 5):
    random.seed(seed)
    
    in_path = Path(input_file)
    if not in_path.exists():
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Reading taxonomy from {tax_file}...")
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > rank_idx:
                tax_map[parts[0]] = parts[rank_idx]

    print(f"Reading {input_file}...")
    open_fn = gzip.open if in_path.suffix == '.gz' else open
    
    records = []
    with open_fn(input_file, "rt") as f:
        for rec in SeqIO.parse(f, "fasta"):
            if rec.id in tax_map:
                rec.id = f"{rec.id}|{tax_map[rec.id]}"
                rec.description = ""
                records.append(rec)
        
    random.shuffle(records)
    
    test_size = int(len(records) * test_ratio)
    test_records = records[:test_size]
    train_records = records[test_size:]
    
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    print(f"Writing {len(train_records)} to {train_file}...")
    SeqIO.write(train_records, train_file, "fasta")
    
    print(f"Writing {len(test_records)} to {test_file}...")
    SeqIO.write(test_records, test_file, "fasta")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input FASTA")
    parser.add_argument("--train", type=str, required=True, help="Output Train FASTA")
    parser.add_argument("--test", type=str, required=True, help="Output Test FASTA")
    parser.add_argument("--tax", type=str, required=True, help="Taxonomy file mapping")
    parser.add_argument("--ratio", type=float, default=0.2, help="Test ratio")
    parser.add_argument("--rank_idx", type=int, default=5, help="Taxonomic rank index (2=Phylum, 3=Class, 4=Order, 5=Family, 6=Genus)")
    
    args = parser.parse_args()
    split_fasta(args.input, args.train, args.test, args.tax, args.ratio, rank_idx=args.rank_idx)
