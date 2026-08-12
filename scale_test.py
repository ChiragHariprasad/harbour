import time
import argparse
import random
import psutil
import os
from hierarchical_infer import HierarchicalPipeline

def generate_random_dna(length):
    return ''.join(random.choices("ACGT", k=length))

def run_scalability_test(args):
    print(f"Generating {args.num_queries} synthetic queries for scalability testing...")
    queries = [generate_random_dna(args.seq_len) for _ in range(args.num_queries)]
    
    pipeline = HierarchicalPipeline(marker=args.marker)
    
    print(f"Starting benchmark for {args.num_queries} queries...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    t0 = time.time()
    
    # We batch if needed, but the current pipeline.predict is single sequence.
    # To test scalability, we just loop it (simulate streaming queries)
    for i, seq in enumerate(queries):
        if (i+1) % (max(1, args.num_queries // 10)) == 0:
            print(f"  Processed {i+1}/{args.num_queries}...")
        _ = pipeline.predict(seq)
        
    t_elapsed = time.time() - t0
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    avg_latency = (t_elapsed / args.num_queries) * 1000
    
    print("\n===== SCALABILITY BENCHMARK =====")
    print(f"Total Queries:     {args.num_queries}")
    print(f"Total Time:        {t_elapsed:.2f} s")
    print(f"Avg Latency:       {avg_latency:.2f} ms / query")
    print(f"Throughput:        {args.num_queries / t_elapsed:.2f} queries / s")
    print(f"Peak RAM Overhead: {mem_after - mem_before:.2f} MB")
    
    # Write hardware details
    import platform
    with open("HARDWARE.md", "w") as f:
        f.write("# Hardware & Infrastructure Details\n")
        f.write(f"- **OS**: {platform.system()} {platform.release()}\n")
        f.write(f"- **Processor**: {platform.processor()}\n")
        f.write(f"- **Total RAM**: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
        f.write(f"\n## Scalability Results\n")
        f.write(f"- **Queries processed**: {args.num_queries}\n")
        f.write(f"- **Throughput**: {args.num_queries / t_elapsed:.2f} queries / s\n")
        f.write(f"- **Peak RAM Overhead**: {mem_after - mem_before:.2f} MB\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--marker", type=str, default="16S")
    
    args = parser.parse_args()
    run_scalability_test(args)
