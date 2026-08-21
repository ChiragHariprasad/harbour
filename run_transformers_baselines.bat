@echo off

echo Starting Phase 17 Embed...
python E:\HTRP\phase17_dnaberts_embed.py > E:\paper_revamp\phase17_dnaberts_baseline\embed.log 2>&1
if %errorlevel% neq 0 (
    echo Phase 17 Embed failed! Check embed.log
    exit /b %errorlevel%
)

echo Starting Phase 17 Baseline...
python E:\HTRP\phase17_dnaberts_baseline.py > E:\paper_revamp\phase17_dnaberts_baseline\baseline.log 2>&1
if %errorlevel% neq 0 (
    echo Phase 17 Baseline failed!
    exit /b %errorlevel%
)

echo Starting Phase 18 Embed...
python E:\HTRP\phase18_nt_embed.py > E:\paper_revamp\phase18_nt_baseline\embed.log 2>&1
if %errorlevel% neq 0 (
    echo Phase 18 Embed failed!
    exit /b %errorlevel%
)

echo Starting Phase 18 Baseline...
python E:\HTRP\phase18_nt_baseline.py > E:\paper_revamp\phase18_nt_baseline\baseline.log 2>&1
if %errorlevel% neq 0 (
    echo Phase 18 Baseline failed!
    exit /b %errorlevel%
)

echo All Transformers Baselines Complete!
