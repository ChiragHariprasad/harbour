import os
import json
import hashlib
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections import defaultdict
import torch
import faiss
import time
from sklearn.metrics import classification_report, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
import joblib

from app import encode_sequence, CNNEmbedder, DEVICE, VOCAB_SIZE

base_dir = "E:/HTRP"
ranks = ['class', 'order', 'family', 'genus']
seeds = [42, 43, 44, 45, 46]

TOP_K = 10
Z_NOVELTY_THRESHOLD = 2.0
CONF_THRESHOLD = 0.6

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
    # Mock config hash for provenance
    return hashlib.md5(f"{rank}_{seed}".encode()).hexdigest()

def infer_supcon(seq, model, index, ids, stats):
    X = encode_sequence(seq)
    with torch.no_grad():
        emb = model(X).cpu().numpy()
        
    faiss.normalize_L2(emb)
    # Search for TOP_K + 1 in case of exact matches (though closed set eval sequences were excluded from train_fit)
    D, I = index.search(emb, TOP_K)
    
    hits = ids[I[0]].tolist()
    dists = D[0].tolist()
    
    from collections import Counter
    label, freq = Counter(hits).most_common(1)[0]
    conf = freq / TOP_K
    
    mean_dist = float(dists[0])
    z = (mean_dist - stats["mean"]) / stats["std"]
    
    prediction = label if (conf >= CONF_THRESHOLD and z < Z_NOVELTY_THRESHOLD) else "NOVEL"
    
    return prediction, conf, z


def main():
    print("PHASE 4: Closed-Set Benchmark Execution")
    
    all_predictions = []
    aggregate_results = []
    validation_report = {
        "status": "PASS",
        "errors": [],
        "configurations": {}
    }
    
    os.makedirs(f"{base_dir}/phase4_confusion_matrices", exist_ok=True)
    
    with open(f"{base_dir}/phase4_dataset_manifest.json") as f:
        dataset_manifest = json.load(f)
    
    for rank in ranks:
        for seed in seeds:
            print(f"--- Running Eval: {rank} Seed {seed} ---")
            
            fit_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_fit.fasta"
            eval_fa = f"{base_dir}/assets/HTRP_cache/16S/{rank}/seed_{seed}/train_eval.fasta"
            model_dir = f"{base_dir}/assets/HTRP_models/embedding/16S/{rank}/seed_{seed}"
            index_dir = f"{base_dir}/assets/HTRP_indexes/16S/{rank}/seed_{seed}"
            baseline_path = f"{base_dir}/assets/HTRP_models/baseline/16S/{rank}/seed_{seed}/model.pkl"
            
            manifest_info = dataset_manifest.get(f"{rank}_seed{seed}", {})
            fit_hash = manifest_info.get("fit_hash")
            eval_hash = manifest_info.get("eval_hash")
            
            # Load Eval Sequences
            eval_records = list(SeqIO.parse(eval_fa, "fasta"))
            eval_ids = [r.id for r in eval_records]
            eval_seqs = [str(r.seq).upper() for r in eval_records]
            y_true = [r.id.split('|')[-1] for r in eval_records]
            
            # Load Fit Taxa for validation
            fit_records = list(SeqIO.parse(fit_fa, "fasta"))
            fit_taxa = set(r.id.split('|')[-1] for r in fit_records)
            fit_ids = set(r.id for r in fit_records)
            
            # VALIDATION CHECKS
            if len(set(eval_ids)) != len(eval_ids):
                validation_report["errors"].append(f"{rank}_{seed}: Non-unique eval IDs")
            if not set(y_true).issubset(fit_taxa):
                validation_report["errors"].append(f"{rank}_{seed}: Eval taxa not subset of Fit taxa")
            if len(set(eval_ids) & fit_ids) > 0:
                validation_report["errors"].append(f"{rank}_{seed}: Eval sequences leak into Fit")
                
            # Load SupCon
            model = CNNEmbedder().to(DEVICE)
            model.load_state_dict(torch.load(f"{model_dir}/embedding.pt", map_location=DEVICE))
            model.eval()
            checkpoint_hash = get_hash(f"{model_dir}/embedding.pt")
            
            index = faiss.read_index(f"{index_dir}/index.faiss")
            ids = np.load(f"{index_dir}/ids.npy", allow_pickle=True)
            with open(f"{index_dir}/meta.json") as f:
                stats = json.load(f)
                
            if len(set(eval_ids) & set(ids)) > 0:
                validation_report["errors"].append(f"{rank}_{seed}: Eval sequences leak into FAISS")
                
            # Load Baseline
            from src.baseline import ClassicalBaseline
            baseline = ClassicalBaseline(k=4)
            baseline.load(baseline_path)
            
            # Baseline predictions
            baseline_preds_raw = baseline.predict_with_novelty(eval_seqs, novelty_threshold=0.5)
            
            y_pred_supcon = []
            y_pred_baseline = []
            
            for i, (seq_id, seq, true_taxon, b_pred) in enumerate(zip(eval_ids, eval_seqs, y_true, baseline_preds_raw)):
                # SupCon
                s_pred, s_conf, s_z = infer_supcon(seq, model, index, ids, stats)
                y_pred_supcon.append(s_pred)
                
                # Baseline
                b_taxon = b_pred["predicted_taxon"]
                b_conf = 1.0 - b_pred["novelty_score"]
                y_pred_baseline.append(b_taxon)
                
                # Record Predictions
                all_predictions.append({
                    "experiment_id": f"{rank}_{seed}",
                    "rank": rank,
                    "seed": seed,
                    "model": "SupCon",
                    "sequence_id": seq_id,
                    "true_taxon": true_taxon,
                    "predicted_taxon": s_pred,
                    "confidence": s_conf,
                    "checkpoint_id": "embedding.pt",
                    "checkpoint_hash": checkpoint_hash,
                    "config_hash": get_config_hash(rank, seed),
                    "dataset_version": "2.0.0_strict_dedup",
                    "train_fit_manifest_hash": fit_hash,
                    "train_eval_manifest_hash": eval_hash
                })
                
                all_predictions.append({
                    "experiment_id": f"{rank}_{seed}",
                    "rank": rank,
                    "seed": seed,
                    "model": "ClassicalBaseline",
                    "sequence_id": seq_id,
                    "true_taxon": true_taxon,
                    "predicted_taxon": b_taxon,
                    "confidence": b_conf,
                    "checkpoint_id": "model.pkl",
                    "checkpoint_hash": get_hash(baseline_path),
                    "config_hash": get_config_hash(rank, seed),
                    "dataset_version": "2.0.0_strict_dedup",
                    "train_fit_manifest_hash": fit_hash,
                    "train_eval_manifest_hash": eval_hash
                })
                
            # Metrics
            for m_name, y_pred in [("SupCon", y_pred_supcon), ("ClassicalBaseline", y_pred_baseline)]:
                # Compute metrics
                acc = np.mean(np.array(y_true) == np.array(y_pred))
                
                rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                macro_prec = rep["macro avg"]["precision"]
                macro_rec = rep["macro avg"]["recall"]
                macro_f1 = rep["macro avg"]["f1-score"]
                weighted_f1 = rep["weighted avg"]["f1-score"]
                
                bal_acc = balanced_accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                
                aggregate_results.append({
                    "experiment_id": f"{rank}_{seed}",
                    "rank": rank,
                    "seed": seed,
                    "model": m_name,
                    "accuracy": acc,
                    "macro_precision": macro_prec,
                    "macro_recall": macro_rec,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "balanced_accuracy": bal_acc,
                    "mcc": mcc
                })
                
                # Confusion Matrix
                labels = sorted(list(set(y_true) | set(y_pred)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                cm_df.to_csv(f"{base_dir}/phase4_confusion_matrices/{rank}_seed{seed}_{m_name}.csv")

    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv(f"{base_dir}/phase4_predictions.csv", index=False)
    
    df_results = pd.DataFrame(aggregate_results)
    df_results.to_csv(f"{base_dir}/Phase4_Main_Benchmark_Results.csv", index=False)
    
    if len(validation_report["errors"]) > 0:
        validation_report["status"] = "FAIL"
        
    with open(f"{base_dir}/phase4_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=4)
        
    print("PHASE 4 COMPLETE.")
    print(f"Status: {validation_report['status']}")
    if validation_report["errors"]:
        print("ERRORS:")
        for e in validation_report["errors"]:
            print(f" - {e}")

if __name__ == "__main__":
    main()
