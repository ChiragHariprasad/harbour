import os
import sys
import time
import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class DNASequenceDataset(Dataset):
    def __init__(self, fasta_file):
        self.records = list(SeqIO.parse(fasta_file, "fasta"))
        
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        return self.records[idx].id, str(self.records[idx].seq)

def collate_fn(batch):
    ids = [item[0] for item in batch]
    seqs = [item[1] for item in batch]
    return ids, seqs

def embed_corpus():
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 16
        print(f"Setting torch threads to 4")
        torch.set_num_threads(4)
        batch_size = 32
        num_workers = 4
    else:
        batch_size = 32
        num_workers = 4
        
    print("Loading DNABERT-S model...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-S", trust_remote_code=True, _fast_init=False)
    model = model.to(device)
    model.eval()
    
    fasta_path = "E:/HTRP/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled.fasta"
    output_dir = "E:/paper_revamp/phase17_dnaberts_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = DNASequenceDataset(fasta_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Extracting embeddings for {len(dataset)} sequences...")
    t0 = time.time()
    
    all_ids = []
    all_embeddings = []
    
    for i, (ids, seqs) in enumerate(dataloader):
        # DNABERT-S tokenization
        inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Take the CLS token embedding (index 0) or the hidden_state
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            batch_emb = hidden_states[:, 0, :].cpu().numpy()
            
        all_ids.extend(ids)
        all_embeddings.append(batch_emb)
        
        if i % 10 == 0 and i > 0:
            print(f"Processed {i * batch_size} / {len(dataset)} sequences (Time: {time.time()-t0:.2f}s)", flush=True)
            
    wall_time = time.time() - t0
    
    final_embeddings = np.vstack(all_embeddings)
    
    out_emb_path = os.path.join(output_dir, "corpus_embeddings.npy")
    out_ids_path = os.path.join(output_dir, "corpus_ids.npy")
    
    np.save(out_emb_path, final_embeddings)
    np.save(out_ids_path, np.array(all_ids))
    
    print(f"\nExtraction complete in {wall_time:.2f} seconds.")
    print(f"Saved to {out_emb_path}, shape: {final_embeddings.shape}")
    
    report = f"Total Time: {wall_time:.2f}s\nBatch Size: {batch_size}\nDevice: {device}\n"
    with open(os.path.join(output_dir, "embed_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    embed_corpus()
