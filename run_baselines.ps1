$ErrorActionPreference = "Stop"

$train_fasta = "assets\HTRP_cache\16S\genus\train.fasta"
$test_fasta = "assets\HTRP_cache\16S\genus\test.fasta"

Write-Host "Running Baselines..."

# 1. K-mer ML Baseline
Write-Host "1. Kmer+ML Baseline"
python baselines\kmer_ml.py --train_fasta $train_fasta --test_fasta $test_fasta --out baseline_kmer.json

# 2. BLAST Baseline (Limited due to API)
Write-Host "2. NCBI BLAST Online Baseline (50 queries)"
python baselines\blast_api.py --test_fasta $test_fasta --limit 50 --out baseline_blast.json

Write-Host "Baselines complete!"
