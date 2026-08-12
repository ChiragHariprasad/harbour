$markers = @("16S", "18S", "CO1", "FITS", "PITS", "trnL", "Vert12S")

foreach ($marker in $markers) {
    Write-Host "========================================"
    Write-Host "Processing $marker..."
    Write-Host "========================================"
    
    $fasta = "assets\HTRP_cache\$marker\family\subsampled.fasta"
    $modelDir = "assets\HTRP_models\embedding\$marker"
    $indexDir = "assets\HTRP_indexes\$marker\family"
    
    if (-Not (Test-Path $fasta)) {
        Write-Host "Fasta not found: $fasta. Skipping."
        continue
    }
    
    # Train model (2 epochs for speed)
    Write-Host "Training model for $marker..."
    python train.py --data $fasta --output_dir $modelDir --epochs 2 --loss supcon
    
    # Build index
    Write-Host "Building index for $marker..."
    python build_index.py --fasta $fasta --model_path "$modelDir\embedding.pt" --output_dir $indexDir --marker $marker --rank family
}
Write-Host "All missing markers processed."
