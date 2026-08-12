# 7. Future Scope

The HTRP architecture and DEEP extensions provide a massive foundation for advancing automated metagenomic classification. While multi-modal consensus and unsupervised novelty clustering have been actively integrated into the current pipeline, several key areas present opportunities for future expansion:

## 1. Cloud Scalability and Distributed Inference
As sample volumes grow, transitioning the current synchronous inference API into a distributed, event-driven microservices architecture (e.g., using Kafka and Kubernetes) will allow horizontal scaling of the `infer_batch` workloads across multiple GPU nodes.

## 2. Long-Read Integration (PacBio/Nanopore)
While current tests validate the system on short amplicon sequences, extending the embedding logic and k-mer routing models to handle full genome assemblies or long-read fragments natively would allow the system to function as a pan-genomic search engine, bypassing the need for alignment-based mapping entirely.

## 3. Fungal and Viral Specialized Pipelines
The pipeline currently excels at eukaryotic and prokaryotic markers (16S, 18S, CO1, 12S). Future work will focus on training robust embedding models and creating comprehensive FAISS indexes specifically optimized for FITS/PITS (fungi) and viral amplicons, which often exhibit higher sequence divergence.

## 4. Continuous Learning and Automated Database Updates
To keep pace with the rapidly expanding public reference databases, a continuous learning MLOps pipeline should be established. This would involve automatically detecting distribution shifts, periodically re-training the marker-specific CNNs on novel authentic sequences, and versioning the FAISS indexes without requiring manual intervention, while strictly tracking `model_version`, `index_version`, and `reference_db_version`.
