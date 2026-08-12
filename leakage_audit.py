import os
import glob
import numpy as np
from Bio import SeqIO
from collections import defaultdict
import scipy.sparse as sp
import time
from sklearn.feature_extraction.text import CountVectorizer

def kmer_count_sparse(seqs, k=6):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), binary=True)
    mat = vectorizer.fit_transform(seqs)
    return mat, vectorizer

def get_taxa(header):
    parts = header.split('|')
    if len(parts) > 1:
        return parts[-1]
    return "Unknown"

def exact_overlap(train_file, test_file):
    train_seqs = set(str(r.seq) for r in SeqIO.parse(train_file, "fasta"))
    test_seqs = set(str(r.seq) for r in SeqIO.parse(test_file, "fasta"))
    overlap = train_seqs.intersection(test_seqs)
    return len(overlap)

def taxa_overlap(train_file, test_file):
    train_taxa = set(get_taxa(r.description) for r in SeqIO.parse(train_file, "fasta"))
    test_taxa = set(get_taxa(r.description) for r in SeqIO.parse(test_file, "fasta"))
    overlap = train_taxa.intersection(test_taxa)
    return len(overlap), len(train_taxa), len(test_taxa)

def similarity_leakage(train_file, test_file):
    print("Loading sequences for similarity check...")
    train_recs = list(SeqIO.parse(train_file, "fasta"))
    test_recs = list(SeqIO.parse(test_file, "fasta"))
    
    train_seqs = [str(r.seq) for r in train_recs]
    test_seqs = [str(r.seq) for r in test_recs]
    
    print("Building train sparse matrix...")
    train_mat, vectorizer = kmer_count_sparse(train_seqs)
    
    print("Building test sparse matrix...")
    test_mat = vectorizer.transform(test_seqs)
    
    train_sizes = np.array(train_mat.sum(axis=1)).flatten()
    test_sizes = np.array(test_mat.sum(axis=1)).flatten()
    
    leakage_99 = 0
    leakage_95 = 0
    leakage_90 = 0
    
    chunk_size = 5000
    for i in range(0, test_mat.shape[0], chunk_size):
        test_chunk = test_mat[i:i+chunk_size]
        intersection = test_chunk.dot(train_mat.T) # Keep it sparse
        
        for j in range(intersection.shape[0]):
            test_idx = i + j
            row = intersection.getrow(j)
            if row.nnz == 0:
                continue
                
            # indices of train sequences that have >0 intersection
            train_indices = row.indices
            inter_vals = row.data
            
            jaccards = inter_vals / (train_sizes[train_indices] + test_sizes[test_idx] - inter_vals + 1e-9)
            max_jaccard = np.max(jaccards)
            
            if max_jaccard >= 0.94:
                leakage_99 += 1
            if max_jaccard >= 0.73:
                leakage_95 += 1
            if max_jaccard >= 0.53:
                leakage_90 += 1
                
    return leakage_99, leakage_95, leakage_90

def faiss_leakage(test_file, ids_npy):
    test_ids = set(r.id for r in SeqIO.parse(test_file, "fasta"))
    if not os.path.exists(ids_npy):
        return -1
    faiss_ids = set(np.load(ids_npy, allow_pickle=True))
    overlap = test_ids.intersection(faiss_ids)
    return len(overlap)

if __name__ == "__main__":
    results = []
    
    ranks = ['family', 'genus', 'order', 'class']
    
    for rank in ranks:
        train_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/train.fasta"
        test_file = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/test.fasta"
        ids_npy = f"E:/HTRP/assets/HTRP_indexes/16S/{rank}/ids.npy"
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            continue
            
        print(f"=== Auditing Rank: {rank} ===")
        t0 = time.time()
        
        exact = exact_overlap(train_file, test_file)
        print(f"Exact Duplicate Leakage: {exact}")
        
        taxa_over, t_train, t_test = taxa_overlap(train_file, test_file)
        print(f"Taxonomic Leakage: {taxa_over} (Train: {t_train}, Test: {t_test})")
        
        faiss_leak = faiss_leakage(test_file, ids_npy)
        print(f"FAISS Index Leakage: {faiss_leak}")
        
        leak_99, leak_95, leak_90 = similarity_leakage(train_file, test_file)
        print(f"Similarity Leakage (approx Jaccard) - 99%: {leak_99}, 95%: {leak_95}, 90%: {leak_90}")
        
        print(f"Completed in {time.time() - t0:.1f}s\n")
        
        results.append(f"{rank},{exact},{taxa_over},{faiss_leak},{leak_99},{leak_95},{leak_90}")

    with open("E:/HTRP/leakage_report.csv", "w") as f:
        f.write("rank,exact_duplicates,taxonomic_overlap,faiss_leakage,sim_99,sim_95,sim_90\n")
        for line in results:
            f.write(line + "\n")
    print("Saved E:/HTRP/leakage_report.csv")
