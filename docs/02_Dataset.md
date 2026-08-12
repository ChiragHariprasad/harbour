# 2. Dataset and Feature Engineering

## Data Sources
The foundational dataset utilized to train the hierarchical classifiers consists of consensus marker gene sequences (primarily 16S rRNA for prokaryotes) aggregated from several curated public repositories:
- **SILVA**: Comprehensive ribosomal RNA databases.
- **PR2**: Protist Ribosomal Reference database.
- **BOLD / UNITE**: Eukaryotic and fungal marker databases.
- **NCBI GenBank / SRA**: Environmental metagenomic reads.

## Cleaning and Preprocessing
The raw FASTA files underwent rigorous preprocessing to ensure model stability and reproducibility:
1. **Quality Filtering**: Sequences containing excessive ambiguous bases (e.g., `N`) or falling outside the expected length distribution for the 16S amplicon were stripped.
2. **Taxonomic Alignment**: Sequence headers were parsed to extract strict taxonomic lineage strings (`Phylum;Class;Order;Family;Genus;Species`). Records with incomplete taxonomy at the target rank were dropped.
3. **True Open-Set Splitting (Hold-out Protocol)**: Instead of randomly splitting sequences, the dataset was subjected to a strict 90/10 taxonomic hold-out protocol (`split_data.py`). Entire unique taxonomic lineages (e.g., specific Genera) were deterministically excluded from the training set and placed exclusively in the test set. This simulates a real environmental sample where the model encounters organisms structurally absent from its reference library.

## Feature Extraction (Tokenization)
1. **Deterministic Path**: The initial Jaccard matcher extracts overlapping **21-mers** from the sequence to compute set intersections with the reference database.
2. **ML Cascade Path**: The CNN pipeline does not use arbitrary string matching. Sequences are tokenized using an overlapping **k-mer window ($K=4$)**. 
3. **Deterministic Seeding**: To resolve prior instability (Bug #2), the $4^4 = 256$ dimensional vocabulary mapping is generated using strict alphabetical sorting and global random seeds, guaranteeing hardware-agnostic deterministic tokenization.

## Weighting and Imbalance Handling
Metagenomic databases are notoriously imbalanced (e.g., *Proteobacteria* vastly outnumber rare phyla). 
- **Supervised Contrastive Learning (SupCon)**: Rather than applying static class weights (which often over-penalize majority classes), the system utilizes a SupCon loss function. This naturally handles imbalance by optimizing the embedding space distances—pulling sequences of the same taxon together and pushing different taxa apart, regardless of raw class frequency.
- **Distance Normalization**: During inference, Z-scores are normalized using the specific empirical mean and standard deviation of the FAISS index at that specific rank, natively calibrating for density variations in the latent space.
