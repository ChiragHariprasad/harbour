import time
import argparse
import json
from Bio.Blast import NCBIWWW, NCBIXML
from Bio import SeqIO
import sys

def run_blast_baseline(args):
    print(f"Reading test data from {args.test_fasta}...")
    seqs = []
    labels = []
    
    for rec in SeqIO.parse(args.test_fasta, "fasta"):
        parts = rec.id.split("|")
        labels.append(parts[-1] if len(parts) > 1 else rec.id)
        seqs.append(str(rec.seq))
        
        if args.limit > 0 and len(seqs) >= args.limit:
            break
            
    print(f"Loaded {len(seqs)} queries. Initiating online BLAST (this may take a while)...")
    
    results = []
    times = []
    
    for i, (seq, true_label) in enumerate(zip(seqs, labels)):
        print(f"Query {i+1}/{len(seqs)}...", end=" ")
        sys.stdout.flush()
        
        t0 = time.time()
        try:
            # query nt database using megablast
            result_handle = NCBIWWW.qblast("blastn", "nt", seq, megablast=True, hitlist_size=1)
            blast_record = NCBIXML.read(result_handle)
            t_elapsed = time.time() - t0
            
            top_hit = "Unknown"
            eval_score = 1.0
            
            if len(blast_record.alignments) > 0:
                alignment = blast_record.alignments[0]
                top_hit = alignment.title
                if len(alignment.hsps) > 0:
                    eval_score = alignment.hsps[0].expect
                    
            print(f"Hit: {top_hit[:30]} (E={eval_score}) [{t_elapsed:.1f}s]")
            
            results.append({
                "true_label": true_label,
                "pred_hit": top_hit,
                "e_value": eval_score,
                "time": t_elapsed
            })
            
        except Exception as e:
            t_elapsed = time.time() - t0
            print(f"Error: {str(e)} [{t_elapsed:.1f}s]")
            results.append({
                "true_label": true_label,
                "pred_hit": "ERROR",
                "e_value": 1.0,
                "time": t_elapsed
            })
            
        times.append(t_elapsed)
        
        # NCBI limits requests, sleep to be polite
        if i < len(seqs) - 1:
            time.sleep(3)
            
    avg_latency = sum(times) / len(times) * 1000
    print(f"\nCompleted BLAST evaluation. Average latency: {avg_latency:.2f} ms")
    
    out = {
        "baseline": "NCBI_BLAST",
        "total_queries": len(seqs),
        "average_latency_ms": avg_latency,
        "results": results
    }
    
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fasta", type=str, required=True)
    parser.add_argument("--limit", type=int, default=10, help="Max queries to run (API is slow)")
    parser.add_argument("--out", type=str, default="baseline_blast.json")
    
    args = parser.parse_args()
    run_blast_baseline(args)
