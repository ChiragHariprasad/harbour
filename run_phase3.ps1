$marker = "16S"
$cacheDir = "assets\HTRP_cache\$marker\family"
$modelDir = "assets\HTRP_models\embedding\$marker"
$indexDir = "assets\HTRP_indexes\$marker\family"
$baselineModel = "assets\HTRP_models\baseline\$marker\model.pkl"

$inputFasta = "$cacheDir\subsampled.fasta"
$taxFile = "$cacheDir\subsampled_taxonomy.txt"
$trainFasta = "$cacheDir\train.fasta"
$testFasta = "$cacheDir\test.fasta"

# Write-Host "1. Splitting Data..."
# python split_data.py --input $inputFasta --train $trainFasta --test $testFasta --tax $taxFile --ratio 0.2

Write-Host "2. Training Classical Baseline..."
python src\baseline.py --train $trainFasta --test $testFasta --out $baselineModel --metrics_out "baseline_metrics_$marker.json"

Write-Host "3. Training HARBOUR (SupCon)..."
python train.py --data $trainFasta --output_dir $modelDir --epochs 3 --loss supcon --metrics_out "train_metrics_supcon_$marker.json"

Write-Host "4. Building FAISS index (SupCon)..."
python build_index.py --fasta $trainFasta --model_path "$modelDir\embedding.pt" --output_dir $indexDir --marker $marker --rank family

Write-Host "5. Benchmarking SupCon..."
python benchmark.py --test $testFasta --marker $marker --baseline_model $baselineModel --out "benchmark_supcon.json"

Write-Host "6. Training HARBOUR (InfoNCE)..."
python train.py --data $trainFasta --output_dir $modelDir --epochs 3 --loss infonce --metrics_out "train_metrics_infonce_$marker.json"

Write-Host "7. Building FAISS index (InfoNCE)..."
python build_index.py --fasta $trainFasta --model_path "$modelDir\embedding.pt" --output_dir $indexDir --marker $marker --rank family

Write-Host "8. Benchmarking InfoNCE..."
python benchmark.py --test $testFasta --marker $marker --baseline_model $baselineModel --out "benchmark_infonce.json"

Write-Host "Done!"
