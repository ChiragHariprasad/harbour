import os
import argparse
from Bio import Entrez
from Bio import SeqIO
import time

def fetch_marine_edna(args):
    Entrez.email = args.email
    print(f"Searching NCBI for marine environmental eDNA 16S samples (limit={args.limit})...")
    
    # Query nucleotide database for marine environmental 16S
    search_query = "marine metagenome[ORGN] AND 16S[All Fields]"
    
    try:
        handle = Entrez.esearch(db="nucleotide", term=search_query, retmax=args.limit)
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"Found {record['Count']} records, downloading top {len(id_list)}...")
        
        if not id_list:
            print("No records found.")
            return
            
        fetch_handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
        data = fetch_handle.read()
        fetch_handle.close()
        
        out_path = args.out
        with open(out_path, "w") as f:
            f.write(data)
            
        print(f"Successfully downloaded to {out_path}.")
        
        # Verify
        count = len(list(SeqIO.parse(out_path, "fasta")))
        print(f"Verified {count} sequences in the FASTA file.")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", type=str, default="researcher@example.com", help="NCBI requires an email")
    parser.add_argument("--limit", type=int, default=1000, help="Number of sequences to fetch")
    parser.add_argument("--out", type=str, default="case_study_marine.fasta", help="Output file")
    
    args = parser.parse_args()
    fetch_marine_edna(args)
