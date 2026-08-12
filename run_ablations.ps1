$ErrorActionPreference = "Stop"

$train_fasta = "assets\HTRP_cache\16S\genus\train.fasta"
$test_fasta = "assets\HTRP_cache\16S\genus\test.fasta"

Write-Host "Running Ablation Studies..."

# 1. Full Model
Write-Host "1. Full Model"
python eval_harness.py --train_fasta $train_fasta --test_fasta $test_fasta --out eval_full.json

# 2. Without Jaccard
Write-Host "2. Without Jaccard"
python eval_harness.py --train_fasta $train_fasta --test_fasta $test_fasta --no_jaccard --out eval_no_jaccard.json

# 3. Without Novelty Detection
Write-Host "3. Without Novelty"
python eval_harness.py --train_fasta $train_fasta --test_fasta $test_fasta --no_novelty --out eval_no_novelty.json

# 4. Without Hierarchy (Always run to Genus)
Write-Host "4. Without Hierarchy"
python eval_harness.py --train_fasta $train_fasta --test_fasta $test_fasta --no_hierarchy --out eval_no_hierarchy.json

Write-Host "Ablations complete! Check eval_*.json for results."
