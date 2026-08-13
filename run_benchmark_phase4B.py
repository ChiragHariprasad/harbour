import os
import json
import hashlib
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import Counter
import torch
import faiss
import time
from sklearn.metrics import classification_report, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
import concurrent.futures

from app import encode_sequence, CNNEmbedder, DEVICE
from src.baseline_corrected import ClassicalBaselineCorrected

base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

TOP_K = 10
Z_NOVELTY_THRESHOLD = 2.0
CONF_THRESHOLD = 0.6

tax_file = f"{base_dir}/assets/HTRP_cache_LEAKED_P2/16S/family/subsampled_taxonomy.txt"

def get_hash(file_path):
    if not os.path.exists(file_path): return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def get_config_hash(rank, seed):
    return hashlib.md5(f"{rank}_{seed}".encode()).hexdigest()

def load_taxonomy_map(target_rank: str, source_rank="family"):
    rank_map = {"class": 3, "order": 4, "family": 5, "genus": 6}
    target_idx = rank_map.get(target_rank.lower(), 5)
    source_idx = rank_map.get(source_rank.lower(), 5)
    
    tax_map = {}
    with open(tax_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > max(target_idx, source_idx):
                tax_map[parts[0]] = parts[target_idx]
                tax_map[parts[source_idx]] = parts[target_idx]
    return tax_map

def infer_supcon(seq, model, index, ids, stats, tax_map):
    X = encode_sequence(seq)
    with torch.no_grad():
        emb = model(X).cpu().numpy()
        
    faiss.normalize_L2(emb)
    D, I = index.search(emb, TOP_K)
    
    hits = ids[I[0]].tolist()
    dists = D[0].tolist()
    
    # Translate FAISS hits (which are Family labels) up to Target Rank
    translated_hits = [tax_map.get(hit, hit) for hit in hits]
    
    label, freq = Counter(translated_hits).most_common(1)[0]
    conf = freq / TOP_K
    
    mean_dist = float(dists[0])
    z = (mean_dist - stats["mean"]) / stats["std"]
    
    prediction = label if (conf >= CONF_THRESHOLD and z < Z_NOVELTY_THRESHOLD) else "NOVEL"
    
    return prediction, conf, z

def evaluate_configuration(args):
    rank, seed, dataset_manifest = args
    print(f"[{rank} | seed {seed}] Starting Eval...")
    
    fit_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
    eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
    model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
    index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
    
    manifest_info = dataset_manifest.get(f"{rank}_seed{seed}", {})
    fit_hash = manifest_info.get("fit_hash")
    eval_hash = manifest_info.get("eval_hash")
    
    tax_map = load_taxonomy_map(target_rank=rank)
    
    # 1. Load Eval
    eval_records = list(SeqIO.parse(eval_fa, "fasta"))
    eval_ids_raw = [r.id.split('|')[0] for r in eval_records]
    eval_seqs = [str(r.seq).upper() for r in eval_records]
    y_true = [tax_map.get(id_raw, "Unknown") for id_raw in eval_ids_raw]
    
    # 2. Retrain Corrected Baseline
    baseline = ClassicalBaselineCorrected(k=4)
    baseline.train(fit_fa, tax_file, rank)
    baseline_preds_raw = baseline.predict_with_novelty(eval_seqs, novelty_threshold=0.5)
    
    # 3. Load SupCon & FAISS
    model = CNNEmbedder().to(DEVICE)
    model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
    model.eval()
    checkpoint_hash = get_hash(f"{model_dir}/embedding.pt")
    
    index = faiss.read_index(f"{index_dir}/index.faiss")
    faiss_ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
    with open(f"{index_dir}/meta.json") as f:
        stats = json.load(f)
        
    predictions = []
    
    for i, (seq_id, seq, true_t, b_pred) in enumerate(zip(eval_ids_raw, eval_seqs, y_true, baseline_preds_raw)):
        # SupCon
        s_pred, s_conf, s_z = infer_supcon(seq, model, index, faiss_ids, stats, tax_map)
        
        # Baseline
        b_taxon = b_pred["predicted_taxon"]
        b_conf = 1.0 - b_pred["novelty_score"]
        
        predictions.append({
            "experiment_id": f"{rank}_{seed}",
            "rank": rank,
            "seed": seed,
            "model": "SupCon",
            "sequence_id": seq_id,
            "true_taxon": true_t,
            "predicted_taxon": s_pred,
            "confidence": s_conf,
            "checkpoint_id": "embedding.pt",
            "checkpoint_hash": checkpoint_hash,
            "config_hash": get_config_hash(rank, seed),
            "dataset_version": "2.0.0_strict_dedup",
            "train_fit_manifest_hash": fit_hash,
            "train_eval_manifest_hash": eval_hash
        })
        
        predictions.append({
            "experiment_id": f"{rank}_{seed}",
            "rank": rank,
            "seed": seed,
            "model": "ClassicalBaseline",
            "sequence_id": seq_id,
            "true_taxon": true_t,
            "predicted_taxon": b_taxon,
            "confidence": b_conf,
            "checkpoint_id": "model.pkl",
            "checkpoint_hash": "DynamicallyTrained",
            "config_hash": get_config_hash(rank, seed),
            "dataset_version": "2.0.0_strict_dedup",
            "train_fit_manifest_hash": fit_hash,
            "train_eval_manifest_hash": eval_hash
        })
        
    # Metrics
    results = []
    cm_dfs = {}
    
    # Validation checks
    errors = []
    if len(set(eval_ids_raw)) != len(eval_ids_raw):
        errors.append(f"Non-unique eval IDs")
        
    for m_name in ["SupCon", "ClassicalBaseline"]:
        preds_model = [p["predicted_taxon"] for p in predictions if p["model"] == m_name]
        
        acc = np.mean(np.array(y_true) == np.array(preds_model))
        rep = classification_report(y_true, preds_model, output_dict=True, zero_division=0)
        
        results.append({
            "experiment_id": f"{rank}_{seed}",
            "rank": rank,
            "seed": seed,
            "model": m_name,
            "accuracy": acc,
            "macro_precision": rep["macro avg"]["precision"],
            "macro_recall": rep["macro avg"]["recall"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "weighted_f1": rep["weighted avg"]["f1-score"],
            "balanced_accuracy": balanced_accuracy_score(y_true, preds_model),
            "mcc": matthews_corrcoef(y_true, preds_model)
        })
        
        labels = sorted(list(set(y_true) | set(preds_model)))
        cm = confusion_matrix(y_true, preds_model, labels=labels)
        cm_dfs[f"{rank}_seed{seed}_{m_name}"] = pd.DataFrame(cm, index=labels, columns=labels)
        
    return predictions, results, cm_dfs, errors

def main():
    print("PHASE 4B: Corrected Rebenchmark")
    
    with open(f"{base_dir}/master Q1 details/Phase_4_Benchmark/phase4_dataset_manifest.json") as f:
        dataset_manifest = json.load(f)
        
    tasks = [(r, s, dataset_manifest) for r in ranks for s in seeds]
    workers = max(1, os.cpu_count() - 1)
    
    all_preds = []
    all_results = []
    all_errors = []
    
    os.makedirs(f"{base_dir}/phase4B_confusion_matrices", exist_ok=True)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for preds, res, cms, errors in executor.map(evaluate_configuration, tasks):
            all_preds.extend(preds)
            all_results.extend(res)
            all_errors.extend(errors)
            
            for cm_name, cm_df in cms.items():
                cm_df.to_csv(f"{base_dir}/phase4B_confusion_matrices/{cm_name}.csv")
                
    df_preds = pd.DataFrame(all_preds)
    df_preds.to_csv(f"{base_dir}/phase4B_predictions.csv", index=False)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f"{base_dir}/Phase4B_Baseline_Results.csv", index=False)
    
    report = {
        "status": "PASS" if not all_errors else "FAIL",
        "errors": all_errors
    }
    with open(f"{base_dir}/phase4B_validation_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    provenance = {
        "baseline_model_class": "MultinomialNB",
        "baseline_script": "src/baseline_corrected.py",
        "supcon_inference_translation": "Enabled (Family -> Target Rank via Taxonomy Map)"
    }
    with open(f"{base_dir}/phase4B_baseline_provenance.json", "w") as f:
        json.dump(provenance, f, indent=4)
        
    print("PHASE 4B REBENCHMARK COMPLETE!")
    
if __name__ == "__main__":
    main()
