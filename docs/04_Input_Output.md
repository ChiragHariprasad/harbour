# 4. Input / Output and API Endpoints

The HTRP system is deployed as a FastAPI web service, allowing seamless integration with bioinformatics pipelines or direct web-browser usage.

## Endpoints

### 1. Web UI (GET `/`)
- **Description**: Serves the main D.E.E.P (DNA Environmental Evaluation Pipeline) user interface or a minimal HTML fallback form if `index.html` is absent.
- **Output**: HTML page allowing file uploads and text area sequence submission.

### 2. Single Sequence Inference (POST `/infer`)
- **Description**: Accepts a single raw nucleotide sequence and routes it through the hierarchical pipeline.
- **Parameters** (`application/x-www-form-urlencoded`):
  - `sequence` (string): The raw DNA sequence (e.g., `ACGT...`).
  - `marker` (string): The genetic marker (e.g., `16S`, `18S`, `CO1`, `auto`). If `auto` is selected, the internal `MarkerDetector` dynamically classifies the sequence before routing it to the appropriate sub-model.
- **Output Format (JSON)**:
  ```json
  {
    "run_id": "uuid-string",
    "query": "query_sequence",
    "path": "ml",
    "prediction": "NOVEL",
    "results": {
      "phylum": {
         "prediction": "Known",
         "confidence": 0.85,
         "z_score": 0.5,
         "votes": [...]
      },
      "class": {
         "prediction": "NOVEL",
         "z_score": 1.8
      }
    },
    "timing": {"deterministic": 0.01, "ml": 0.45, "total": 0.46}
  }
  ```

### 3. Batch Inference (POST `/infer_batch`)
- **Description**: Accepts a full `.fasta` file upload for high-throughput batch classification.
- **Parameters** (`multipart/form-data`):
  - `fasta` (file): A standard FASTA file upload.
  - `marker` (string): The genetic marker constraint.
- **Output Format (JSON)**:
  Returns a `summary` dictionary aggregating counts of `known`, `novel`, `corrupted`, and the specific pipeline path (`deterministic` vs `ml_based`) utilized, followed by the complete list of individual sequence results.

## Input Processing
All sequences undergo pre-flight Quality Control (`src/qc.py`) before entering the classifier. Sequences containing excessive ambiguities (e.g., `N` characters) are immediately short-circuited and flagged as `CORRUPTED` to prevent garbage-in/garbage-out degradation of the ML embeddings.
