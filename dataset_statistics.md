# HTRP Dataset & Evaluation Detailed Statistics

This document provides the authoritative numbers required for the Methods section.

---

## 1. Per-Rank Taxonomy Statistics (Deduplicated Source, N=100,000)

These statistics were computed directly from the deduplicated `subsampled.fasta` cross-referenced with `subsampled_taxonomy.txt`.

| Rank | Unique Taxa | Sequences with Valid Taxon | Min Seqs/Taxon | Median Seqs/Taxon | Max Seqs/Taxon | Mean ± Stdev |
|------|-------------|---------------------------|----------------|-------------------|----------------|--------------|
| **Class** | 306 | 88,566 | 1 | 42 | 5,853 | 289.4 ± 725.5 |
| **Order** | 1,256 | 96,706 | 1 | 16 | 2,293 | 77.0 ± 179.7 |
| **Family** | 5,142 | 99,836 | 1 | 4 | 175 | 19.4 ± 36.8 |
| **Genus** | 17,730 | 97,712 | 1 | 2 | 168 | 5.5 ± 14.2 |

> **Note:** "Sequences with Valid Taxon" < 100,000 for some ranks because certain records carry `NA` at that taxonomic level (e.g., 11,434 sequences lack a valid Class annotation).

---

## 2. Sequence Length Statistics (bp)

Computed over all 100,000 sequences in the source FASTA.

| Statistic | Value (bp) |
|-----------|-----------|
| **N** | 100,000 |
| **Min** | 76 |
| **Q1** | 252 |
| **Median** | 254 |
| **Mean** | 301.3 |
| **Q3** | 417 |
| **Max** | 546 |
| **Stdev** | 94.0 |

---

## 3. Per-Configuration Split Counts (All 20 Rank × Seed Cells)

All 20 configurations share the same source of 100,000 raw records, deduplicated to 58,910 unique sequences. Holdout is always performed at the **Family** rank. The train/val/test split varies per seed due to the stochastic taxonomic holdout.

| Config | Rank | Seed | Source | Dedup | N_train | N_val | N_test | Holdout Rank |
|--------|------|------|--------|-------|---------|-------|--------|--------------|
| class_seed42 | class | 42 | 100,000 | 58,910 | 46,978 | 6,497 | 5,435 | family |
| class_seed43 | class | 43 | 100,000 | 58,910 | 46,855 | 5,982 | 6,073 | family |
| class_seed44 | class | 44 | 100,000 | 58,910 | 47,842 | 5,708 | 5,360 | family |
| class_seed45 | class | 45 | 100,000 | 58,910 | 46,318 | 6,100 | 6,492 | family |
| class_seed46 | class | 46 | 100,000 | 58,910 | 46,589 | 6,482 | 5,839 | family |
| order_seed42 | order | 42 | 100,000 | 58,910 | 46,978 | 6,497 | 5,435 | family |
| order_seed43 | order | 43 | 100,000 | 58,910 | 46,855 | 5,982 | 6,073 | family |
| order_seed44 | order | 44 | 100,000 | 58,910 | 47,842 | 5,708 | 5,360 | family |
| order_seed45 | order | 45 | 100,000 | 58,910 | 46,318 | 6,100 | 6,492 | family |
| order_seed46 | order | 46 | 100,000 | 58,910 | 46,589 | 6,482 | 5,839 | family |
| family_seed42 | family | 42 | 100,000 | 58,910 | 46,978 | 6,497 | 5,435 | family |
| family_seed43 | family | 43 | 100,000 | 58,910 | 46,855 | 5,982 | 6,073 | family |
| family_seed44 | family | 44 | 100,000 | 58,910 | 47,842 | 5,708 | 5,360 | family |
| family_seed45 | family | 45 | 100,000 | 58,910 | 46,318 | 6,100 | 6,492 | family |
| family_seed46 | family | 46 | 100,000 | 58,910 | 46,589 | 6,482 | 5,839 | family |
| genus_seed42 | genus | 42 | 100,000 | 58,910 | 46,978 | 6,497 | 5,435 | family |
| genus_seed43 | genus | 43 | 100,000 | 58,910 | 46,855 | 5,982 | 6,073 | family |
| genus_seed44 | genus | 44 | 100,000 | 58,910 | 47,842 | 5,708 | 5,360 | family |
| genus_seed45 | genus | 45 | 100,000 | 58,910 | 46,318 | 6,100 | 6,492 | family |
| genus_seed46 | genus | 46 | 100,000 | 58,910 | 46,589 | 6,482 | 5,839 | family |

> **Key observation:** The train/val/test partition is identical across ranks for a given seed (e.g., seed 42 always produces 46,978/6,497/5,435 regardless of evaluation rank). This is because the holdout is always at the Family level; the rank column only determines the taxonomic resolution at which the model is trained and evaluated.

---

## 4. Confidence–Distance Landscape (Per-Sequence Records)

**The authoritative per-sequence file is:** [`phase6_final_predictions.csv`](file:///e:/HTRP/phase6_final_predictions.csv) (51.0 MB, 156,796 rows).

### Schema
Each row contains per-sequence decision data:

| Column | Description |
|--------|-------------|
| `experiment_id` | e.g. `class_42` |
| `rank` | Evaluation rank |
| `seed` | Random seed |
| `sequence_id` | NCBI accession |
| `true_status_at_rank` | Ground truth: `KNOWN` or `NOVEL` |
| `supcon_predicted_status` | HARBOUR decision: `KNOWN` or `NOVEL` |
| `distance` | L2 distance to nearest FAISS neighbor |
| `z_score` | Empirical z-score of that distance |
| `confidence` | k-NN voting confidence (fraction of TOP_K agreeing) |
| `threshold_z` | Z threshold used (frozen: **1.0**) |
| `threshold_conf` | Confidence threshold used (frozen: **0.5**) |
| `baseline_probability` | Random Forest baseline probability |
| `baseline_predicted_status` | Baseline decision |
| `true_class/order/family/genus` | Full taxonomic lineage |

### Sample Rows
```
AF463459.1  KNOWN  KNOWN  z=0.301  conf=1.0  dist=0.078  (Class: Insecta)
GQ488240.1  KNOWN  KNOWN  z=-0.466 conf=1.0  dist=0.011  (Class: Arachnida)
GQ488255.1  KNOWN  NOVEL  z=1.933  conf=1.0  dist=0.222  (Class: Arachnida) ← False Novel
DQ367454.1  KNOWN  NOVEL  z=1.039  conf=0.6  dist=0.143  (Class: Harpellomycetes) ← True, z > 1.0
```

This file contains the exact `(Z, confidence, ground-truth, decision)` quadruples for all 156,796 evaluation records across all 20 configurations.

---

## 5. Preprocessing Exclusion Counts

### Pipeline Attrition Funnel

| Stage | Count | Notes |
|-------|-------|-------|
| **Raw subsampled records** | 100,000 | Starting file: `subsampled.fasta` |
| **Exact sequence duplicates removed** | 41,090 | Detected via `split_data.py` SHA-256 sequence dedup |
| **Post-dedup unique sequences** | 58,910 | Used for all downstream splits |
| **Sequences containing ambiguous bases (N)** | 2,136 | Present in the 100k set; **not excluded** — the k-mer tokenizer skips individual N-containing k-mers |
| **Sequences < 100 bp** | 8 | Present but **not filtered** at this stage |
| **Sequences > 5,000 bp** | 0 | None present |
| **Sequences missing taxonomy** | 0 | All 100k records have entries in the taxonomy file |
| **Final usable sequences** | **58,910** | After deduplication only |

> **Important clarification on upstream preprocessing:**
> The 100,000-record `subsampled.fasta` was itself the product of an upstream data preparation step (outside this repository) that pulled from SILVA and other databases, applied taxonomy-completeness filtering, length filtering, and ambiguous-base filtering, and then subsampled to 100k records. The exact attrition counts from *that* upstream step are **not recorded in this repository**. Within this repository, the only filtering applied is exact sequence-level deduplication (100,000 → 58,910).
