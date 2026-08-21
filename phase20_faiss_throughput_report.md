# FAISS HNSW32 Throughput Benchmark

## Hardware Configuration
- **Processor**: Intel64 Family 6 Model 197 Stepping 2, GenuineIntel (16 physical / 16 logical cores)
- **RAM**: 31.50 GB
- **FAISS Config**: OpenMP multi-threading enabled (16 threads)

## Index Specifications
- **Type**: HNSW32 (M=32, efConstruction=200)
- **Search Param**: efSearch=64
- **Size in Memory**: 33.64 MB
- **Corpus Size**: 44978 reference vectors (128-d)

## Performance Results

| Mode | Batch Size | p50 Latency | p95 Latency | Throughput (QPS) |
|------|------------|-------------|-------------|------------------|
| Single | 1 | 0.0297 ms | 0.0466 ms | 30,409 |
| Batched | 7435 | 30.99 ms | 34.99 ms | 239,465 |