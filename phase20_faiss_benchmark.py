import os
import time
import psutil
import platform
import numpy as np
import faiss
import torch
from Bio import SeqIO

# For extracting real queries
from app import encode_sequence, CNNEmbedder, DEVICE

def measure_hardware():
    print("--- Hardware Specification ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical Threads: {psutil.cpu_count(logical=True)}")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print("------------------------------")

def main():
    measure_hardware()
    
    # Configure FAISS multi-threading
    num_threads = psutil.cpu_count(logical=True)
    faiss.omp_set_num_threads(num_threads)
    print(f"FAISS OpenMP Threads set to: {num_threads}")
    
    rank = "family"
    seed = 42
    
    # 1. Load frozen index
    index_path = f"E:/HTRP/assets/HTRP_indexes/16S/{rank}/seed_{seed}/index.faiss"
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"\nLoading index from {index_path} ({index_size_mb:.2f} MB)")
    
    index = faiss.read_index(index_path)
    
    # Ensure HNSW and efSearch=64
    if isinstance(index, faiss.IndexHNSWFlat) or isinstance(index, faiss.IndexHNSW):
        index.hnsw.efSearch = 64
        print(f"Index is HNSW. efSearch set to {index.hnsw.efSearch}.")
    else:
        print("Warning: Index is not explicitly HNSWFlat. Proceeding anyway.")
        
    print(f"Total vectors in index: {index.ntotal}")
    
    # 2. Prepare Realistic Queries
    print("\nPreparing realistic queries from test partition...")
    cache_dir = f"E:/HTRP/assets/HTRP_cache/16S/{rank}/seed_{seed}"
    test_records = list(SeqIO.parse(f"{cache_dir}/test.fasta", "fasta"))
    # Limit to 10k for speed if there are more, but test is likely smaller (~2.5k)
    # Let's also use train_eval to get a good 10k queries
    eval_records = list(SeqIO.parse(f"{cache_dir}/train_eval.fasta", "fasta"))
    all_records = test_records + eval_records
    
    model_dir = f"E:/HTRP/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    
    query_embs = []
    batch_size = 128
    for i in range(0, len(all_records), batch_size):
        batch_seqs = [str(r.seq) for r in all_records[i:i+batch_size]]
        X_batch = torch.cat([encode_sequence(seq) for seq in batch_seqs], dim=0)
        with torch.no_grad():
            emb = model(X_batch.to(DEVICE)).cpu().numpy()
        query_embs.append(emb)
        
    query_embs = np.vstack(query_embs).astype('float32')
    faiss.normalize_L2(query_embs)
    num_queries = len(query_embs)
    print(f"Generated {num_queries} L2-normalized 128-d queries.")
    
    # Warmup
    _ = index.search(query_embs[:100], 10)
    
    # 3. Single-Query Measurement (Latency focused)
    print("\n--- Single-Query Benchmarking ---")
    latencies = []
    for i in range(num_queries):
        q = query_embs[i:i+1]
        t0 = time.perf_counter()
        index.search(q, 10)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # ms
        
    latencies = np.array(latencies)
    p50_sq = np.percentile(latencies, 50)
    p95_sq = np.percentile(latencies, 95)
    total_time_sq = np.sum(latencies) / 1000
    qps_sq = num_queries / total_time_sq
    
    print(f"p50 Latency: {p50_sq:.4f} ms")
    print(f"p95 Latency: {p95_sq:.4f} ms")
    print(f"Throughput:  {qps_sq:.2f} QPS")
    
    # 4. Batched-Query Measurement (Throughput focused)
    print(f"\n--- Batched-Query Benchmarking (Batch Size={num_queries}) ---")
    # For a stable batch throughput, run a few times
    batch_qps_list = []
    batch_times = []
    for _ in range(10): # 10 iterations
        t0 = time.perf_counter()
        index.search(query_embs, 10)
        t1 = time.perf_counter()
        batch_times.append((t1 - t0) * 1000)
        batch_qps_list.append(num_queries / (t1 - t0))
        
    p50_bq = np.percentile(batch_times, 50)
    p95_bq = np.percentile(batch_times, 95)
    mean_qps_bq = np.mean(batch_qps_list)
    
    print(f"p50 Latency (per batch): {p50_bq:.2f} ms")
    print(f"p95 Latency (per batch): {p95_bq:.2f} ms")
    print(f"Throughput: {mean_qps_bq:.2f} QPS")
    
    # 5. Output small table as CSV/Markdown
    report = f"""
# FAISS HNSW32 Throughput Benchmark

## Hardware Configuration
- **Processor**: {platform.processor()} ({psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical cores)
- **RAM**: {psutil.virtual_memory().total / (1024**3):.2f} GB
- **FAISS Config**: OpenMP multi-threading enabled ({num_threads} threads)

## Index Specifications
- **Type**: HNSW32 (M=32, efConstruction=200)
- **Search Param**: efSearch=64
- **Size in Memory**: {index_size_mb:.2f} MB
- **Corpus Size**: {index.ntotal} reference vectors (128-d)

## Performance Results

| Mode | Batch Size | p50 Latency | p95 Latency | Throughput (QPS) |
|------|------------|-------------|-------------|------------------|
| Single | 1 | {p50_sq:.4f} ms | {p95_sq:.4f} ms | {qps_sq:,.0f} |
| Batched | {num_queries} | {p50_bq:.2f} ms | {p95_bq:.2f} ms | {mean_qps_bq:,.0f} |

"""
    
    with open("E:/HTRP/phase20_faiss_throughput_report.md", "w") as f:
        f.write(report.strip())
        
    print("\nReport saved to E:/HTRP/phase20_faiss_throughput_report.md")

if __name__ == "__main__":
    main()
