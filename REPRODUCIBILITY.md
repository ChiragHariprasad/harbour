# Reproducibility Statement

This document details the exact reproducibility status of the HARBOUR analytical pipeline.

## Reproducibility Tiers

To prevent overstating the end-to-end reproducibility of missing upstream components (like raw multi-node model training), we explicitly separate the workflow into two tiers:

| Component | Reproducibility status | Description |
|-----------|------------------------|-------------|
| Phases 1–4 | Provenance / frozen inputs | These phases constitute the upstream model generation and are provided as static provenance artifacts (checkpoint hashes, embedding definitions, dataset hashes) and frozen model outputs. |
| Phase 5 | Reproducible | Threshold selection from validation distributions. |
| Phase 6 | Reproducible | Final open-set evaluation predictions. |
| Phase 7 | Reproducible | DBSCAN structural robustness on test embeddings. |
| Phase 8 | Intentionally skipped | Skipped due to missing paired biological metadata in the raw dataset. |
| Phase 9 | Reproducible | Embedding/taxonomy structural correlation and zero-shot evaluations. |
| Phase 10 | Reproducible | Sequence-level diagnostic error analysis and homonym checks. |
| Phase 11 | Reproducible | B=1,000 paired bootstrap statistical significance testing. |
| Final artifacts | Hash-verified | Deterministic execution verified against published hash tolerances. |

## How to Reproduce the Analytical Pipeline (Phases 5-11)

The complete evaluation pipeline (Phases 5 through 11) is fully reproducible from the provided frozen predictions and embeddings.

### 1. Environment Setup
You may use Docker or Conda to ensure system-level and Python-level dependency lock:
**Using Docker:**
```bash
docker build -t harbour-eval .
docker run -it harbour-eval
```
**Using Conda:**
```bash
conda env create -f environment.yml
conda activate harbour-eval
```

### 2. Execution
Run the master orchestrator script:
```bash
python reproduce_all.py
```

This script will automatically:
1. Verify the Python environment.
2. Verify all cryptographic input hashes defined in `dataset_manifest.json`.
3. Sequentially execute the analytical scripts without modifying frozen parameters.
4. Export a final `reproduction_manifest.json` documenting the newly generated artifacts.
5. Check those generated artifacts against the published baseline to guarantee perfect reproduction.
