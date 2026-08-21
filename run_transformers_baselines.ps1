$ErrorActionPreference = "Stop"

Write-Host "Starting Phase 17 Embed..."
python E:\HTRP\phase17_dnaberts_embed.py 2>&1 | Out-File -FilePath E:\paper_revamp\phase17_dnaberts_baseline\embed.log

Write-Host "Starting Phase 17 Baseline..."
python E:\HTRP\phase17_dnaberts_baseline.py 2>&1 | Out-File -FilePath E:\paper_revamp\phase17_dnaberts_baseline\baseline.log

Write-Host "Starting Phase 18 Embed..."
python E:\HTRP\phase18_nt_embed.py 2>&1 | Out-File -FilePath E:\paper_revamp\phase18_nt_baseline\embed.log

Write-Host "Starting Phase 18 Baseline..."
python E:\HTRP\phase18_nt_baseline.py 2>&1 | Out-File -FilePath E:\paper_revamp\phase18_nt_baseline\baseline.log

Write-Host "All Transformers Baselines Complete!"
