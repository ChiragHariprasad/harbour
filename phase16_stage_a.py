import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from Bio import SeqIO

BASE_DIR = "E:/HTRP"
OUTPUT_BASE = "E:/paper_revamp/phase16_kraken2_baseline"
WSL_KRAKEN2_BUILD = "$HOME/kraken2_bin/kraken2-build"

def load_taxonomy_map(target_rank: str, source_rank="family"):
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(target_rank.lower(), 5)
    source_idx = rank_map.get(source_rank.lower(), 5)
    
    tax_map = {}
    tax_file = f"{BASE_DIR}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
    return tax_map

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def run_stage_a(rank="class", seed=42):
    print(f"--- Kraken2 Stage A: {rank} seed {seed} ---")
    t0 = time.time()
    
    cache_dir = f"{BASE_DIR}/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    db_dir = f"{OUTPUT_BASE}/db_{rank}_{seed}"
    wsl_db_dir = db_dir.replace("E:/", "/mnt/e/")
    
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    os.makedirs(f"{db_dir}/taxonomy", exist_ok=True)
    
    tax_map = load_taxonomy_map(target_rank=rank)
    fit_fa = f"{cache_dir}/train_fit.fasta"
    fit_records = list(SeqIO.parse(fit_fa, "fasta"))
    
    unique_labels = set()
    for r in fit_records:
        acc = r.id.split('|')[0]
        unique_labels.add(tax_map.get(acc, "Unknown"))
        
    unique_labels = sorted(list(unique_labels))
    label_to_taxid = {label: i+2 for i, label in enumerate(unique_labels)}
    label_to_taxid["Root"] = 1
    
    with open(f"{db_dir}/taxonomy/nodes.dmp", "w", newline='\n') as f:
        f.write("1\t|\t1\t|\tno rank\t|\n")
        for label, taxid in label_to_taxid.items():
            if taxid == 1: continue
            f.write(f"{taxid}\t|\t1\t|\t{rank}\t|\n")
            
    with open(f"{db_dir}/taxonomy/names.dmp", "w", newline='\n') as f:
        f.write("1\t|\tRoot\t|\t\t|\tscientific name\t|\n")
        for label, taxid in label_to_taxid.items():
            if taxid == 1: continue
            f.write(f"{taxid}\t|\t{label}\t|\t\t|\tscientific name\t|\n")
            
    seq_out = []
    print("\n[CONSTRAINT-9 SPOT CHECK]")
    spot_checked = 0
    for r in fit_records:
        acc = r.id.split('|')[0]
        label = tax_map.get(acc, "Unknown")
        taxid = label_to_taxid[label]
        r.id = f"kraken:taxid|{taxid}|{acc}|{label}"
        r.description = ""
        seq_out.append(r)
        if spot_checked < 5:
            print(f"Accession: {acc} -> Label mapped to: {label} -> TaxID: {taxid} -> Header: {r.id}")
            spot_checked += 1
            
    print("-------------------------")
    
    # Write to a file, then kraken2-build --add-to-library
    temp_fa = f"{db_dir}/temp.fasta"
    wsl_temp_fa = wsl_db_dir + "/temp.fasta"
    SeqIO.write(seq_out, temp_fa, "fasta")
    
    print("\nAdding to library...")
    add_cmd = f"wsl bash -c '{WSL_KRAKEN2_BUILD} --add-to-library {wsl_temp_fa} --db {wsl_db_dir}'"
    subprocess.run(add_cmd, shell=True, check=True)
    os.remove(temp_fa)
    
    print("\nBuilding Kraken2 Database...")
    # use /usr/bin/time -v to get max memory
    build_cmd = f"wsl bash -c '/usr/bin/time -v {WSL_KRAKEN2_BUILD} --build --db {wsl_db_dir} --threads 16'"
    result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
    
    print("\n--- BUILD STDOUT/STDERR ---")
    print(result.stderr)
    
    # Parse memory from time output
    peak_mem_kb = 0
    for line in result.stderr.split('\n'):
        if "Maximum resident set size (kbytes)" in line:
            peak_mem_kb = int(line.split(':')[-1].strip())
            
    wall_time = time.time() - t0
    disk_bytes = get_dir_size(db_dir)
    
    print("\n--- RESOURCE USAGE ---")
    print(f"Wall-clock Time: {wall_time:.2f} seconds")
    print(f"Peak Memory: {peak_mem_kb / 1024 / 1024:.2f} GB")
    print(f"Disk Size: {disk_bytes / 1024 / 1024 / 1024:.2f} GB")

if __name__ == "__main__":
    run_stage_a()
