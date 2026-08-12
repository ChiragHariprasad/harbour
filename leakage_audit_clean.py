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

def audit_split(train_file, val_file, test_file):
    train_seqs = set()
    train_taxa = set()
    for r in SeqIO.parse(train_file, "fasta"):
        train_seqs.add(str(r.seq))
        train_taxa.add(get_taxa(r.description))
        
    val_seqs = set()
    val_taxa = set()
    if os.path.exists(val_file):
        for r in SeqIO.parse(val_file, "fasta"):
            val_seqs.add(str(r.seq))
            val_taxa.add(get_taxa(r.description))

    test_seqs = set()
    test_taxa = set()
    for r in SeqIO.parse(test_file, "fasta"):
        test_seqs.add(str(r.seq))
        test_taxa.add(get_taxa(r.description))
        
    exact_tr_te = len(train_seqs.intersection(test_seqs))
    exact_tr_va = len(train_seqs.intersection(val_seqs))
    exact_va_te = len(val_seqs.intersection(test_seqs))
    
    tax_tr_te = len(train_taxa.intersection(test_taxa))
    tax_tr_va = len(train_taxa.intersection(val_taxa))
    tax_va_te = len(val_taxa.intersection(test_taxa))
    
    return exact_tr_te, exact_tr_va, exact_va_te, tax_tr_te, tax_tr_va, tax_va_te

if __name__ == "__main__":
    results = []
    ranks = ['class', 'order', 'family', 'genus']
    
    for rank in ranks:
        train_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/train.fasta"
        val_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/val.fasta"
        test_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/test.fasta"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            continue
            
        print(f"=== Re-Auditing Cleaned Split: {rank} ===")
        t0 = time.time()
        
        e_tr_te, e_tr_va, e_va_te, t_tr_te, t_tr_va, t_va_te = audit_split(train_file, val_file, test_file)
        
        print(f"Exact Leakage -> Train-Test: {e_tr_te}, Train-Val: {e_tr_va}, Val-Test: {e_va_te}")
        print(f"Taxa Leakage  -> Train-Test: {t_tr_te}, Train-Val: {t_tr_va}, Val-Test: {t_va_te}")
        print(f"Completed in {time.time() - t0:.1f}s\n")
        
        results.append({
            'rank': rank,
            'exact_tr_te': e_tr_te,
            'taxa_tr_te': t_tr_te
        })
