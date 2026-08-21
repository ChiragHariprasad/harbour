import os
import torch
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from Bio import SeqIO
from app import encode_sequence, CNNEmbedder, DEVICE

def main():
    print("Loading Phase 6 predictions...")
    preds_file = "E:/HTRP/phase6_final_predictions.csv"
    df = pd.read_csv(preds_file)
    
    # Filter to Family rank, seed 42
    df_sub = df[(df['rank'] == 'family') & (df['seed'] == 42)].copy()
    print(f"Found {len(df_sub)} predictions for family/seed_42.")
    
    # Determine HARBOUR Outcome
    def get_outcome(row):
        # We need to compute novelty based on the finalized threshold logic.
        # But wait, phase6_final_predictions.csv already has 'supcon_predicted_status'.
        true_novel = (row['true_status_at_rank'] == 'NOVEL')
        pred_novel = (row['supcon_predicted_status'] == 'NOVEL')
        
        if true_novel and pred_novel: return "Correct Novel"
        elif true_novel and not pred_novel: return "False Known"
        elif not true_novel and not pred_novel: return "Correct Known"
        elif not true_novel and pred_novel: return "False Novel"
        return "Unknown"
        
    df_sub['harbour_outcome'] = df_sub.apply(get_outcome, axis=1)
    
    # Get Top 15 families, set rest to "Other"
    top_families = df_sub['true_taxon_at_rank'].value_counts().nlargest(15).index.tolist()
    df_sub['plot_family'] = df_sub['true_taxon_at_rank'].apply(lambda x: x if x in top_families else 'Other')
    
    # Load sequences to extract embeddings
    print("Loading sequences...")
    cache_dir = "E:/HTRP/assets/HTRP_cache/16S/family/seed_42"
    records = []
    for split in ['train_eval.fasta', 'val.fasta', 'test.fasta']:
        fpath = os.path.join(cache_dir, split)
        if os.path.exists(fpath):
            records.extend(list(SeqIO.parse(fpath, "fasta")))
            
    seq_dict = {r.id.split('|')[0]: str(r.seq) for r in records}
    
    print("Loading CNN-SupCon model...")
    model_dir = "E:/HTRP/assets/HTRP_models/embedding/16S/family/seed_42"
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(model_dir, "embedding.pt"), map_location=DEVICE))
    model.eval()
    
    # Ensure order matches df_sub
    query_ids = df_sub['sequence_id'].tolist()
    missing = [qid for qid in query_ids if qid not in seq_dict]
    if missing:
        print(f"Warning: {len(missing)} query IDs missing from FASTA files! Removing them from plot.")
        df_sub = df_sub[~df_sub['sequence_id'].isin(missing)]
        query_ids = df_sub['sequence_id'].tolist()
        
    print(f"Extracting 128-d embeddings for {len(query_ids)} sequences...")
    embeddings = []
    
    # Process in batches for speed
    batch_size = 128
    for i in range(0, len(query_ids), batch_size):
        batch_ids = query_ids[i:i+batch_size]
        batch_seqs = [seq_dict[qid] for qid in batch_ids]
        
        # encode_sequence usually takes a single string. If it doesn't support batched, we loop:
        X_batch = torch.cat([encode_sequence(seq) for seq in batch_seqs], dim=0)
        
        with torch.no_grad():
            emb = model(X_batch.to(DEVICE)).cpu().numpy()
        embeddings.append(emb)
        
    embeddings = np.vstack(embeddings)
    
    print("L2 normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    print("Running UMAP (n_neighbors=15, min_dist=0.1, metric='cosine')...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    df_sub['umap_x'] = embedding_2d[:, 0]
    df_sub['umap_y'] = embedding_2d[:, 1]
    
    # Save CSV
    out_csv = "E:/HTRP/phase19_umap_family_seed42.csv"
    df_sub.to_csv(out_csv, index=False)
    print(f"Saved coordinates to {out_csv}")
    
    # Plot
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Panel (i) Ground Truth Family
    sns.scatterplot(
        data=df_sub, x='umap_x', y='umap_y', hue='plot_family',
        palette='tab20', s=15, alpha=0.7, edgecolor=None, ax=axes[0]
    )
    axes[0].set_title('UMAP by Ground Truth Family (Top 15)', fontsize=12, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8, markerscale=1.5)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    
    # Panel (ii) HARBOUR Decision Outcome
    outcome_palette = {
        "Correct Known": "#2ca02c", # Green
        "Correct Novel": "#1f77b4", # Blue
        "False Known": "#d62728", # Red
        "False Novel": "#ff7f0e"  # Orange
    }
    sns.scatterplot(
        data=df_sub, x='umap_x', y='umap_y', hue='harbour_outcome',
        palette=outcome_palette, s=15, alpha=0.7, edgecolor=None, ax=axes[1]
    )
    axes[1].set_title('UMAP by HARBOUR Decision Outcome', fontsize=12, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, markerscale=2)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    
    plt.tight_layout()
    out_png = "E:/HTRP/phase19_umap_family_seed42.png"
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {out_png}")
    print("Done!")

if __name__ == "__main__":
    main()
