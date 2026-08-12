$marker = "16S"
$baseCacheDir = "assets\HTRP_cache\$marker"
$baseModelDir = "assets\HTRP_models\embedding\$marker"
$baseIndexDir = "assets\HTRP_indexes\$marker"
$taxFile = "$baseCacheDir\family\subsampled_taxonomy.txt"
$inputFasta = "$baseCacheDir\family\subsampled.fasta"

# Define the ranks and their column index in the taxonomy file
$ranks = @(
    @{ Name="phylum"; Index=2 },
    @{ Name="class"; Index=3 },
    @{ Name="order"; Index=4 },
    @{ Name="family"; Index=5 },
    @{ Name="genus"; Index=6 }
)

Write-Host "========================================="
Write-Host "Starting Hierarchical Cascade Training"
Write-Host "========================================="

foreach ($rank in $ranks) {
    $rankName = $rank.Name
    $rankIdx = $rank.Index
    
    Write-Host "`n--- Processing Rank: $rankName (Index: $rankIdx) ---"
    
    $cacheDir = "$baseCacheDir\$rankName"
    $modelDir = "$baseModelDir\$rankName"
    $indexDir = "$baseIndexDir\$rankName"
    
    $trainFasta = "$cacheDir\train.fasta"
    $testFasta = "$cacheDir\test.fasta"
    
    Write-Host "1. Splitting Data for $rankName..."
    python split_data.py --input $inputFasta --train $trainFasta --test $testFasta --tax $taxFile --ratio 0.2 --rank_idx $rankIdx
    
    Write-Host "2. Training HARBOUR (SupCon) for $rankName..."
    python train.py --data $trainFasta --output_dir $modelDir --epochs 3 --loss supcon --metrics_out "$modelDir\train_metrics.json"
    
    Write-Host "3. Building FAISS index for $rankName..."
    python build_index.py --fasta $trainFasta --model_path "$modelDir\embedding.pt" --output_dir $indexDir --marker $marker --rank $rankName
}

Write-Host "`n========================================="
Write-Host "Cascade Training Complete!"
Write-Host "========================================="
