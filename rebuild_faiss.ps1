$marker = "16S"
$baseCacheDir = "assets\HTRP_cache\$marker"
$baseModelDir = "assets\HTRP_models\embedding\$marker"
$baseIndexDir = "assets\HTRP_indexes\$marker"

$ranks = @("class", "order", "family", "genus")
$seeds = @(42, 43, 44, 45, 46)

Write-Host "========================================="
Write-Host "Rebuilding FAISS indexes for $marker (5 seeds)"
Write-Host "========================================="

foreach ($rank in $ranks) {
    foreach ($seed in $seeds) {
        Write-Host "`n--- Indexing Rank: $rank, Seed: $seed ---"
        
        $cacheDir = "$baseCacheDir\$rank\seed_$seed"
        
        # The model was trained only on seed 42 theoretically, or at the rank level. 
        # For evaluation diagnosis we'll just use whatever model exists at the rank level.
        $modelDir = "$baseModelDir\$rank"
        
        $indexDir = "$baseIndexDir\$rank\seed_$seed"
        
        $trainFasta = "$cacheDir\train.fasta"
        $modelPath = "$modelDir\embedding.pt"
        
        if (-Not (Test-Path $trainFasta)) {
            Write-Host "Missing $trainFasta"
            continue
        }
        
        if (-Not (Test-Path $modelPath)) {
            Write-Host "Missing $modelPath"
            continue
        }
        
        python build_index.py --fasta $trainFasta --model_path $modelPath --output_dir $indexDir --marker $marker --rank $rank
    }
}

Write-Host "`n========================================="
Write-Host "Indexing Complete!"
Write-Host "========================================="
