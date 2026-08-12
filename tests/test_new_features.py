import os
import io
from fastapi.testclient import TestClient
from Bio import SeqIO
from app import app, CACHE_ROOT, get_index

client = TestClient(app)

def test_clustering_stats():
    print("--- Running Clustering Stats on Authentic Marine eDNA ---")
    fasta_path = "case_study_marine.fasta"
    if not os.path.exists(fasta_path):
        print(f"{fasta_path} not found.")
        return

    # Load first 25 sequences to keep the test fast
    records = list(SeqIO.parse(fasta_path, "fasta"))[:25]
    
    # We will format them with marker="auto" to test detection
    fasta_content = "\n".join([f">{rec.id}\n{str(rec.seq)}" for rec in records])
    
    response = client.post(
        "/infer_batch",
        data={"marker": "16S"},
        files={"fasta": ("test.fasta", io.BytesIO(fasta_content.encode()), "text/plain")}
    )
    
    if response.status_code != 200:
        print("Error:", response.text)
        return
        
    data = response.json()
    summary = data["summary"]
    clusters = data["clustering"]
    
    print(f"Total Sequences: {summary['total']}")
    print(f"Known: {summary['known']}")
    print(f"Novel: {summary['novel']}")
    print("Other predictions:")
    from collections import Counter
    preds = Counter(r.get("prediction") for r in data["results"])
    print(preds)
    
    print(f"\nClustering Results:")
    if not clusters:
        print("No clusters formed.")
    for cluster_res in clusters:
        if "clusters" in cluster_res:
            print(f"Marker: {cluster_res.get('marker', 'unknown')}")
            print(f"  Candidates: {cluster_res['candidate_count']}")
            print(f"  Clusters formed: {cluster_res['cluster_count']}")
            print(f"  Noise (isolated): {cluster_res['noise_count']}")
            for c in cluster_res["clusters"]:
                print(f"    - Cluster {c['cluster_id']}: size={c['size']}, z_score={c['mean_z_score']}, affinity={c.get('taxonomic_affinity')}")

def test_consensus_stats():
    print("\n--- Running Multi-Marker Consensus Stats ---")
    
    # Create a synthetic fasta with sample IDs and multiple markers
    # We'll extract a few sequences from the cache to ensure we get known hits
    
    fasta_content = """>sample=S001 marker=16S id=read1
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>sample=S001 marker=16S id=read2
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGA
>sample=S001 marker=16S id=read3
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGC
>sample=S002 marker=16S id=read4
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
>sample=S002 marker=16S id=read5
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCG
"""
    response = client.post(
        "/infer_batch",
        data={"marker": "auto"},
        files={"fasta": ("multi.fasta", io.BytesIO(fasta_content.encode()), "text/plain")}
    )
    
    if response.status_code != 200:
        print("Error:", response.text)
        return
        
    data = response.json()
    consensus_results = data.get("consensus", [])
    
    print(f"Consensus Results: {len(consensus_results)} samples resolved")
    for cons in consensus_results:
        print(f"Sample: {cons['sample_id']}")
        print(f"  Resolved Rank: {cons['resolved_rank']}")
        print(f"  Taxonomy: {cons['consensus_taxonomy']}")
        print(f"  Agreements: {cons['agreements']}")
        if cons['conflicts']:
            print(f"  Conflicts detected: {len(cons['conflicts'])}")
        else:
            print("  No conflicts.")

if __name__ == "__main__":
    test_clustering_stats()
    test_consensus_stats()
