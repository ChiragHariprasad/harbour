"""
Extract detailed dataset statistics for the HTRP research paper.
Computes:
1. Per-rank taxonomy stats (unique taxa, sequence count, min/median/max per taxon)
2. Sequence length stats (min, Q1, median, mean, Q3, max, stdev)
3. All 20 configuration N counts
4. Preprocessing exclusion counts
"""
import os, json, sys
import numpy as np
from collections import Counter, defaultdict
from Bio import SeqIO

base_dir = "E:/HTRP"

# ============================================================
# 1. PER-RANK TAXONOMY STATS from the source taxonomy file
# ============================================================
tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
fasta_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled.fasta"

print("=" * 70)
print("1. PER-RANK TAXONOMY STATISTICS")
print("=" * 70)

# Parse taxonomy file: columns are accession, Kingdom, Phylum, Class, Order, Family, Genus, Species
rank_indices = {"Class": 3, "Order": 4, "Family": 5, "Genus": 6}
tax_data = {}  # accession -> [kingdom, phylum, class, order, family, genus, species]

with open(tax_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 7:
            accession = parts[0]
            tax_data[accession] = parts[1:]  # everything after accession

# Now compute per-rank stats using only accessions present in the deduplicated FASTA
print(f"\nReading FASTA to get deduplicated accession set...")
dedup_accessions = set()
seq_lengths = []

if os.path.exists(fasta_file):
    for rec in SeqIO.parse(fasta_file, "fasta"):
        # IDs may be accession|label format or just accession
        acc = rec.id.split("|")[0]
        dedup_accessions.add(acc)
        seq_lengths.append(len(rec.seq))
    print(f"  Deduplicated sequences in FASTA: {len(dedup_accessions)}")
else:
    print(f"  WARNING: {fasta_file} not found!")

for rank_name, rank_idx in rank_indices.items():
    taxon_counts = Counter()
    for acc in dedup_accessions:
        if acc in tax_data and len(tax_data[acc]) > rank_idx - 1:
            taxon = tax_data[acc][rank_idx - 1]  # -1 because tax_data[acc] starts at Kingdom (index 0)
            if taxon and taxon != "NA":
                taxon_counts[taxon] += 1
    
    counts = list(taxon_counts.values())
    if counts:
        print(f"\n  --- {rank_name} ---")
        print(f"  Unique taxa: {len(taxon_counts)}")
        print(f"  Total sequences with valid {rank_name}: {sum(counts)}")
        print(f"  Sequences per taxon: min={min(counts)}, median={int(np.median(counts))}, max={max(counts)}")
        print(f"  Mean seqs/taxon: {np.mean(counts):.1f}, Stdev: {np.std(counts):.1f}")
    else:
        print(f"\n  --- {rank_name} ---")
        print(f"  No valid data found at rank index {rank_idx}")

# ============================================================
# 2. SEQUENCE LENGTH STATS
# ============================================================
print("\n" + "=" * 70)
print("2. SEQUENCE LENGTH STATISTICS (bp)")
print("=" * 70)

if seq_lengths:
    arr = np.array(seq_lengths)
    print(f"  N sequences: {len(arr)}")
    print(f"  Min:    {np.min(arr)}")
    print(f"  Q1:     {int(np.percentile(arr, 25))}")
    print(f"  Median: {int(np.median(arr))}")
    print(f"  Mean:   {np.mean(arr):.1f}")
    print(f"  Q3:     {int(np.percentile(arr, 75))}")
    print(f"  Max:    {np.max(arr)}")
    print(f"  Stdev:  {np.std(arr):.1f}")

# ============================================================
# 3. ALL 20 CONFIGURATION N COUNTS
# ============================================================
print("\n" + "=" * 70)
print("3. ALL 20 CONFIGURATION N COUNTS")
print("=" * 70)

manifest_file = f"{base_dir}/master Q1 details/dataset_manifest.json"
with open(manifest_file, "r") as f:
    manifest = json.load(f)

print(f"\n  {'Config':<15} {'Rank':<8} {'Seed':<6} {'Source':<8} {'Dedup':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Holdout Rank'}")
print("  " + "-" * 85)

for config_key in sorted(manifest.keys()):
    entry = manifest[config_key]
    counts = entry["counts"]
    # Parse rank and seed from config_key
    parts = config_key.rsplit("_seed", 1)
    rank = parts[0]
    seed = parts[1]
    
    print(f"  {config_key:<15} {rank:<8} {seed:<6} {counts['source_count']:<8} {counts['deduplicated_count']:<8} {counts['train_count']:<8} {counts['val_count']:<8} {counts['test_count']:<8} {entry['holdout_rank']}")

# ============================================================
# 4. PREPROCESSING EXCLUSION COUNTS
# ============================================================
print("\n" + "=" * 70)
print("4. PREPROCESSING / FILTERING COUNTS")
print("=" * 70)

# The pipeline starts from 100,000 records and deduplicates to 58,910
# Let's check the raw FASTA (pre-dedup) if it exists
raw_fasta = fasta_file  # The subsampled.fasta IS the starting point for this repo
print(f"\n  Source file: subsampled.fasta")
print(f"  Source records (pre-dedup): 100,000")
print(f"  Post-dedup records: 58,910")
print(f"  Exact duplicates removed: {100000 - 58910}")

# Check for ambiguous bases in the deduplicated set
if os.path.exists(fasta_file):
    n_with_N = 0
    n_short = 0
    n_long = 0
    n_no_taxonomy = 0
    total_raw = 0
    
    # Re-read to count exclusion-worthy sequences
    for rec in SeqIO.parse(fasta_file, "fasta"):
        total_raw += 1
        seq_str = str(rec.seq).upper()
        if "N" in seq_str:
            n_with_N += 1
        if len(seq_str) < 100:
            n_short += 1
        if len(seq_str) > 5000:
            n_long += 1
        acc = rec.id.split("|")[0]
        if acc not in tax_data:
            n_no_taxonomy += 1
    
    print(f"\n  In the deduplicated 58,910-record set:")
    print(f"    Sequences containing ambiguous bases (N): {n_with_N}")
    print(f"    Sequences < 100 bp: {n_short}")
    print(f"    Sequences > 5000 bp: {n_long}")
    print(f"    Sequences missing from taxonomy file: {n_no_taxonomy}")
    print(f"    Total sequences in final usable set: {total_raw}")

# ============================================================
# 5. CHECK FOR EXISTING PER-SEQUENCE AUDIT CSVs
# ============================================================
print("\n" + "=" * 70)
print("5. EXISTING PER-SEQUENCE PREDICTION/AUDIT FILES")
print("=" * 70)

candidates = [
    "phase6_final_predictions.csv",
    "phase9_nn_purity_predictions.csv",
    "prediction_audit.csv",
    "master Q1 details/prediction_audit.csv",
    "master Q1 details/Phase_4_Benchmark/phase4_predictions.csv",
]

for c in candidates:
    path = f"{base_dir}/{c}"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024*1024)
        # Read header
        with open(path, "r") as f:
            header = f.readline().strip()
        print(f"\n  FOUND: {c} ({size_mb:.1f} MB)")
        print(f"    Columns: {header}")
    else:
        print(f"  NOT FOUND: {c}")

# Check phase6 predictions header and sample
phase6_path = f"{base_dir}/phase6_final_predictions.csv"
if os.path.exists(phase6_path):
    print(f"\n  --- Sample from phase6_final_predictions.csv ---")
    with open(phase6_path, "r") as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"    {line.strip()}")
            else:
                break

print("\n\nDONE.")
