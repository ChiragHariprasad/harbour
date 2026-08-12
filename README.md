# HTRP eDNA Taxonomic Classifier

HTRP is a hierarchical eDNA classification pipeline designed for deep-sea and marine metagenomic samples. It leverages a Deterministic Jaccard filter followed by a 5-stage Machine Learning Cascade (Phylum $\rightarrow$ Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus) using Supervised Contrastive CNN embeddings and FAISS vector search.

This architecture enables **Search by Elimination**, aggressively filtering out novel (unseen) taxa at higher taxonomic ranks to prevent hallucinations at the Genus level.

## Architecture Highlights
- **Deterministic Matcher (Species)**: 100% precision for known taxa using 21-mer Jaccard similarity.
- **ML Cascade**: 5 sequential `CNNEmbedder` models trained independently per rank.
- **Novelty Detection**: Distance-based Z-score anomaly gating ($Z > 1.5$) dynamically flags Out-of-Distribution (OOD) sequences.
- **Speed**: Achieves ~178,000 queries/sec using `IndexIVFFlat`.

---

## Installation

```bash
git clone https://github.com/organization/HTRP.git
cd HTRP
pip install -r requirements.txt
```
*(Requires PyTorch and FAISS. Windows users may need to install `faiss-cpu` or `faiss-gpu` via Conda).*

---

## Usage (Inference)

To classify a single sequence or a FASTA file using the pretrained models:

```python
from hierarchical_infer import HierarchicalPipeline

# Initialize the pipeline (loads FAISS indexes and CNN weights)
pipeline = HierarchicalPipeline(marker="16S")

sequence = "ACGTACGTACGTACGTACGT..."
result = pipeline.predict(sequence)

print(result["prediction"]) 
# Outputs: 'KNOWN_EXACT', 'KNOWN_GENUS', or 'NOVEL_<RANK>'
```

---

## Training (MLOps)

To reproduce the model weights from scratch using your own reference database:

1. **Split Data**:
   ```bash
   python split_data.py --input raw_16S.fasta --train assets/16S/train.fasta --test assets/16S/test.fasta --tax taxonomy.tsv
   ```
2. **Train the Cascade**:
   Use the provided PowerShell automation script to train all 5 rank models sequentially (Phylum to Genus) and build their FAISS indexes:
   ```bash
   powershell .\run_hierarchical.ps1
   ```
3. **Evaluate**:
   ```bash
   python eval_harness.py --train_fasta assets/16S/train.fasta --test_fasta assets/16S/test.fasta
   ```

---

## Data Availability
The reference databases (SILVA, PR2, BOLD, UNITE, MitoFish, GenBank) used for the canonical models in our paper are publicly available. Accession IDs and exact train/test split definitions can be found in the `assets/` metadata.
