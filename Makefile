.PHONY: all test train build_index run_api

# Default target
all: test

# Run tests
test:
	pytest tests/test_smoke.py -v

# Train the model (example)
train:
	python train.py --data assets/HTRP_cache/16S/family/subsampled.fasta --output_dir assets/HTRP_models/embedding/16S

# Build the FAISS index (example)
build_index:
	python build_index.py --fasta assets/HTRP_cache/16S/family/subsampled.fasta --model_path assets/HTRP_models/embedding/16S/embedding.pt --output_dir assets/HTRP_indexes/16S/family --marker 16S --rank family

# Run the API locally
run_api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload
