import os
import numpy as np
from Bio import SeqIO
import time
import csv

def get_taxa(header):
    parts = header.split('|')
    if len(parts) > 1:
        return parts[-1]
    return "Unknown"

def audit_split(train_file, test_file, ids_npy):
    train_seqs = set()
    train_taxa = set()
    for r in SeqIO.parse(train_file, "fasta"):
        train_seqs.add(str(r.seq))
        train_taxa.add(get_taxa(r.description))
        
    test_seqs = set()
    test_taxa = set()
    test_ids = set()
    for r in SeqIO.parse(test_file, "fasta"):
        test_seqs.add(str(r.seq))
        test_taxa.add(get_taxa(r.description))
        test_ids.add(r.id)
        
    exact_leakage = len(train_seqs.intersection(test_seqs))
    taxa_leakage = len(train_taxa.intersection(test_taxa))
    
    faiss_leakage = 0
    if os.path.exists(ids_npy):
        faiss_ids = set(np.load(ids_npy, allow_pickle=True))
        faiss_leakage = len(test_ids.intersection(faiss_ids))
        
    return exact_leakage, taxa_leakage, len(train_taxa), len(test_taxa), faiss_leakage

if __name__ == "__main__":
    results = []
    ranks = ['class', 'order', 'family', 'genus']
    
    for rank in ranks:
        train_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/train.fasta"
        test_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/test.fasta"
        ids_npy = f"E:/HTRP/assets/HTRP_indexes/16S/{rank}/ids.npy"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            continue
            
        print(f"=== Auditing Rank: {rank} ===")
        t0 = time.time()
        
        exact, taxa_over, t_train, t_test, faiss_leak = audit_split(train_file, test_file, ids_npy)
        
        print(f"Exact Duplicate Leakage: {exact}")
        print(f"Taxonomic Leakage: {taxa_over} (Train: {t_train}, Test: {t_test})")
        print(f"FAISS Index Leakage: {faiss_leak}")
        print(f"Completed in {time.time() - t0:.1f}s\n")
        
        results.append({
            'rank': rank,
            'exact_duplicates': exact,
            'taxonomic_overlap': taxa_over,
            'faiss_leakage': faiss_leak,
            'sim_99': "SKIPPED",
            'sim_95': "SKIPPED",
            'sim_90': "SKIPPED"
        })

    with open("E:/HTRP/leakage_report.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank','exact_duplicates','taxonomic_overlap','faiss_leakage','sim_99','sim_95','sim_90'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Saved E:/HTRP/leakage_report.csv")
