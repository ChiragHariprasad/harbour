$marker = "16S"
$baseCacheDir = "assets\HTRP_cache\$marker"
$taxFile = "assets\HTRP_cache_LEAKED_P2\$marker\family\subsampled_taxonomy.txt"
$inputFasta = "assets\HTRP_cache_LEAKED_P2\$marker\family\subsampled.fasta"

$ranks = @("class", "order", "family", "genus")
$seeds = @(42, 43, 44, 45, 46)

Write-Host "========================================="
Write-Host "Generating Clean Splits (5 Seeds) for $marker"
Write-Host "========================================="

foreach ($rank in $ranks) {
    foreach ($seed in $seeds) {
        Write-Host "`n--- Splitting Rank: $rank, Seed: $seed ---"
        
        # Original code used $baseCacheDir\$rank, but now we use seed subdirs
        $cacheDir = "$baseCacheDir\$rank\seed_$seed"
        
        $trainFasta = "$cacheDir\train.fasta"
        $valFasta = "$cacheDir\val.fasta"
        $testFasta = "$cacheDir\test.fasta"
        $manifest = "$cacheDir\split_manifest.json"
        
        $holdoutRank = 5
        
        python split_data.py --input $inputFasta --train $trainFasta --val $valFasta --test $testFasta --tax $taxFile --manifest $manifest --val_ratio 0.1 --test_ratio 0.1 --rank_idx 5 --holdout_rank $holdoutRank --holdout_ratio 0.1 --seed $seed
    }
}

Write-Host "`n========================================="
Write-Host "Splitting Complete!"
Write-Host "========================================="
