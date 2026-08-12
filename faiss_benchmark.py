import faiss
import time
import argparse
import numpy as np
import os
import psutil

def benchmark_faiss(d, n_train, n_test):
    print(f"Generating synthetic embeddings (d={d}, train={n_train}, test={n_test})...")
    np.random.seed(42)
    X_train = np.random.random((n_train, d)).astype('float32')
    X_test = np.random.random((n_test, d)).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(X_train)
    faiss.normalize_L2(X_test)
    
    indices = {
        "IndexFlatIP": faiss.IndexFlatIP(d),
        "IndexHNSWFlat": faiss.IndexHNSWFlat(d, 32),
        "IndexIVFFlat": faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, min(n_train//10, 100), faiss.METRIC_INNER_PRODUCT)
    }
    
    results = {}
    
    for name, index in indices.items():
        print(f"\n--- Testing {name} ---")
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss
        
        t0 = time.time()
        if not index.is_trained:
            index.train(X_train)
        index.add(X_train)
        t_build = time.time() - t0
        
        mem_end = process.memory_info().rss
        mem_overhead_mb = (mem_end - mem_start) / (1024 * 1024)
        
        # Test latency
        t0 = time.time()
        D, I = index.search(X_test, 5)
        t_search = time.time() - t0
        
        throughput = n_test / t_search
        
        print(f"Build Time: {t_build:.2f} s")
        print(f"Memory Overhead: {mem_overhead_mb:.2f} MB")
        print(f"Search Time (for {n_test} queries): {t_search:.2f} s")
        print(f"Throughput: {throughput:.2f} q/s")
        
        results[name] = {
            "build_time_s": t_build,
            "memory_mb": mem_overhead_mb,
            "throughput_qs": throughput
        }
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--train", type=int, default=100000, help="Number of items to index")
    parser.add_argument("--test", type=int, default=10000, help="Number of queries")
    args = parser.parse_args()
    benchmark_faiss(args.d, args.train, args.test)
