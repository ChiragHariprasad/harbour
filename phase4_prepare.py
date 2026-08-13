import os
import glob
import json
import hashlib
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import time
import subprocess
import concurrent.futures
import math
from pathlib import Path

# Fix: Import app dependencies required by train.py and build_index.py globally
# so that the subprocesses can run them
base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

def run_command_in_shell(cmd, cwd=base_dir):
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def get_hash(file_path):
    if not os.path.exists(file_path): return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def prepare_closed_set_split(rank, seed):
    train_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train.fasta"
    fit_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
    eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
    
    # 1. Read sequences
    records = list(SeqIO.parse(train_fa, "fasta"))
    
    # Group by taxon
    taxon_to_recs = defaultdict(list)
    for rec in records:
        taxon = rec.id.split('|')[-1]
        taxon_to_recs[taxon].append(rec)
        
    fit_records = []
    eval_records = []
    
    # Random generator for deterministic sampling
    rng = np.random.RandomState(seed)
    
    eligible_taxa = []
    # Any taxon with 1 sequence goes entirely to fit
    for taxon, recs in taxon_to_recs.items():
        if len(recs) == 1:
            fit_records.extend(recs)
        else:
            eligible_taxa.append(taxon)
            
    # Calculate how many sequences we can take from eligible taxa
    total_eligible = sum(len(taxon_to_recs[t]) for t in eligible_taxa)
    eval_target_size = min(2000, total_eligible // 2)
    
    # We want proportional allocation
    sampled_count = 0
    
    for taxon in eligible_taxa:
        recs = taxon_to_recs[taxon]
        # Always put at least 1 in fit
        # Proportion we can take
        proportion = len(recs) / total_eligible
        allocation = math.ceil(proportion * eval_target_size)
        
        # We can't take all of them (must leave at least 1)
        max_take = len(recs) - 1
        take = min(allocation, max_take)
        
        # If we exceed the target, reduce take
        if sampled_count + take > eval_target_size:
            take = max(0, eval_target_size - sampled_count)
            
        # Shuffle records deterministically
        rng.shuffle(recs)
        eval_records.extend(recs[:take])
        fit_records.extend(recs[take:])
        sampled_count += take

    SeqIO.write(fit_records, fit_fa, "fasta")
    SeqIO.write(eval_records, eval_fa, "fasta")
    
    # Verification Rule: taxa(train_eval) <= taxa(train_fit)
    taxa_fit = set(r.id.split('|')[-1] for r in fit_records)
    taxa_eval = set(r.id.split('|')[-1] for r in eval_records)
    assert taxa_eval.issubset(taxa_fit), f"Taxon leakage in {rank} seed {seed}"
    
    # Verification Rule: no sequence overlaps
    ids_fit = set(r.id for r in fit_records)
    ids_eval = set(r.id for r in eval_records)
    assert len(ids_fit & ids_eval) == 0, f"Sequence overlap in {rank} seed {seed}"
    
    return {
        "N_train_fit": len(fit_records),
        "N_train_eval": len(eval_records),
        "N_taxa_train_fit": len(taxa_fit),
        "N_taxa_train_eval": len(taxa_eval),
        "fit_hash": get_hash(fit_fa),
        "eval_hash": get_hash(eval_fa)
    }

def process_configuration(task):
    rank, seed = task
    print(f"[{rank} | seed {seed}] STARTING...")
    
    # 1. Split dataset
    stats = prepare_closed_set_split(rank, seed)
    
    # 2. Retrain HARBOUR (SupCon) on train_fit.fasta
    fit_fa = f"assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
    model_out = f"assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    metrics_out = f"assets/HTRP_models/embedding/16S/{rank}/seed_{seed}/train_metrics.json"
    
    print(f"[{rank} | seed {seed}] Retraining SupCon...")
    cmd_train = f"python train.py --data {fit_fa} --output_dir {model_out} --loss supcon --epochs 3 --metrics_out {metrics_out}"
    run_command_in_shell(cmd_train)
    
    # 3. Rebuild FAISS index on train_fit.fasta
    index_out = f"assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    model_path = f"{model_out}/embedding.pt"
    
    print(f"[{rank} | seed {seed}] Rebuilding FAISS index...")
    cmd_faiss = f"python build_index.py --fasta {fit_fa} --model_path {model_path} --output_dir {index_out} --marker 16S --rank {rank}"
    run_command_in_shell(cmd_faiss)
    
    # 4. Train Classical Baseline on train_fit.fasta
    baseline_out = f"assets/HTRP_models/baseline/16S/{rank}/seed_{seed}/model.pkl"
    # We will just evaluate inline later, but we can pre-train it here
    cmd_baseline = f"python src/baseline.py --train {fit_fa} --test {fit_fa} --out {baseline_out} --metrics_out {index_out}/baseline_metrics.json"
    print(f"[{rank} | seed {seed}] Training Classical Baseline...")
    run_command_in_shell(cmd_baseline)
    
    print(f"[{rank} | seed {seed}] FINISHED PREPARATION.")
    return rank, seed, stats

def main():
    print("PHASE 4: Closed-Set Benchmark Automation")
    
    tasks = [(rank, seed) for rank in ranks for seed in seeds]
    
    manifest = {}
    
    # Use multiprocessing to train models and build FAISS
    num_workers = max(1, os.cpu_count() // 2)
    print(f"Using {num_workers} parallel workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_configuration, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            rank, seed, stats = future.result()
            manifest[f"{rank}_seed{seed}"] = stats
            
    with open(f"{base_dir}/phase4_dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
        
    print("ALL RETRAINING AND SPLITTING COMPLETE. READY FOR EVALUATION.")

if __name__ == "__main__":
    main()
