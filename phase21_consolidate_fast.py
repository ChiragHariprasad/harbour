"""
Phase 21 — Final Manuscript Consolidation (rewrite)
Produces:
  1. phase21_baseline_table.csv        — mean ± std across 5 seeds, 7 models, 4 ranks
  2. phase21_bootstrap_ci.csv          — B=1000 paired bootstrap 95% CIs
  3. phase21_ablation_table.csv        — copy of existing ablation summary
  4. phase21_findings.md               — interpretive summary

Strategy for paired bootstrap:
  - For each (rank, seed), intersect sequence_ids across all 7 models so every
    bootstrap resample draws the SAME sequences for every model.
  - Pre-extract numpy arrays; never touch pandas inside the inner loop.
  - Use multiprocessing.Pool to parallelise across ranks (4 workers).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             f1_score, matthews_corrcoef)
from multiprocessing import Pool
from functools import partial

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = "E:/HTRP"
REVAMP     = "E:/paper_revamp"
OUT_DIR    = f"{REVAMP}/phase21_final"
RANKS      = ["class", "order", "family", "genus"]
SEEDS      = [42, 43, 44, 45, 46]
MODELS     = ["MultinomialNB", "MSP-full", "VSEARCH", "Kraken2",
              "DNABERT-S", "Nucleotide Transformer", "HARBOUR"]
B          = 1000            # bootstrap replicates
RNG_SEED   = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── metric helper (pure numpy, no pandas) ──────────────────────────────────────
def calc_metrics(y_true_bin, y_pred_bin, y_score, y_true_cls, y_pred_cls):
    """Return dict of {H_score, AUROC, MCC, Novel_F1, Closed_set_F1, A_known, R_novel}."""
    # confusion matrix  (label 0=Known, 1=Novel)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    a_known = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    r_novel = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    h_score = (2 * a_known * r_novel / (a_known + r_novel)
               if (a_known + r_novel) > 0 else 0.0)

    # AUROC
    try:
        auroc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        auroc = float("nan")

    # MCC — build multiclass labels where novel ⇒ "NOVEL"
    # We now expect y_t_mc and y_p_mc to be integers pre-encoded.
    mcc = matthews_corrcoef(y_true_cls, y_pred_cls)

    # Closed-set F1 (known instances only, macro)
    known_mask = (y_true_bin == 0)
    if known_mask.sum() > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            closed_f1 = f1_score(y_true_cls[known_mask],
                                 y_pred_cls[known_mask],
                                 average="macro", zero_division=0)
    else:
        closed_f1 = float("nan")

    # Novel F1 (binary: novel vs known)
    novel_f1 = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    return {"H_score": h_score, "AUROC": auroc, "MCC": mcc,
            "Novel_F1": novel_f1, "Closed_set_F1": closed_f1,
            "A_known": a_known, "R_novel": r_novel}


# ── loaders — each returns (sequence_id[], y_true_cls[], y_pred_cls[],
#                             y_true_bin[], y_pred_bin[], y_score[])
#    ALL as numpy arrays. ───────────────────────────────────────────────────────

_phase6_cache = None
def _phase6():
    global _phase6_cache
    if _phase6_cache is None:
        _phase6_cache = pd.read_csv(f"{BASE}/phase6_final_predictions.csv")
    return _phase6_cache

def load_harbour(rank, seed):
    df = _phase6(); df = df[(df["rank"]==rank)&(df["seed"]==seed)]
    return (df["sequence_id"].values,
            df["true_taxon_at_rank"].values,
            df["supcon_predicted_taxon"].values,
            (df["true_status_at_rank"]=="NOVEL").astype(np.int8).values,
            (df["supcon_predicted_status"]=="NOVEL").astype(np.int8).values,
            df["z_score"].values)

def load_multinomialnb(rank, seed):
    df = _phase6(); df = df[(df["rank"]==rank)&(df["seed"]==seed)]
    return (df["sequence_id"].values,
            df["true_taxon_at_rank"].values,
            df["baseline_predicted_taxon"].values,
            (df["true_status_at_rank"]=="NOVEL").astype(np.int8).values,
            (df["baseline_predicted_status"]=="NOVEL").astype(np.int8).values,
            1.0 - df["baseline_probability"].values)

def load_msp(rank, seed):
    df = pd.read_csv(f"{REVAMP}/phase13_msp_baseline/{rank}/seed_{seed}/predictions.csv")
    return (df["sequence_id"].values,
            df["true_taxon"].values,
            df["predicted_taxon"].values,
            (df["true_status"]=="NOVEL").astype(np.int8).values,
            (df["predicted_status"]=="NOVEL").astype(np.int8).values,
            1.0 - df["msp_score"].values)

def load_vsearch(rank, seed):
    df = pd.read_csv(f"{REVAMP}/phase15_vsearch_baseline/{rank}/seed_{seed}/predictions.csv")
    return (df["sequence_id"].values,
            df["true_taxon"].values,
            df["predicted_taxon"].values,
            (df["true_status"]=="NOVEL").astype(np.int8).values,
            (df["predicted_status"]=="NOVEL").astype(np.int8).values,
            -df["pident"].values)

def load_kraken2(rank, seed):
    df = pd.read_csv(f"{REVAMP}/phase16_kraken2_baseline/db_{rank}_{seed}/predictions.csv")
    gt = _phase6(); gt = gt[(gt["rank"]==rank)&(gt["seed"]==seed)][
        ["sequence_id","true_taxon_at_rank","true_status_at_rank"]]
    df = df.rename(columns={"accession":"sequence_id"})
    df = pd.merge(df, gt, on="sequence_id", how="left")
    return (df["sequence_id"].values,
            df["true_taxon_at_rank"].values,
            df["pred_label"].values,
            (df["true_status_at_rank"]=="NOVEL").astype(np.int8).values,
            df["is_novel"].astype(np.int8).values,
            1.0 - df["confidence"].values)

def _load_embedding_baseline(phase_dir, rank, seed):
    df = pd.read_csv(f"{REVAMP}/{phase_dir}/{rank}/seed_{seed}/predictions.csv")
    gt = _phase6(); gt = gt[(gt["rank"]==rank)&(gt["seed"]==seed)][
        ["sequence_id","true_taxon_at_rank","true_status_at_rank"]]
    df = df.rename(columns={"accession":"sequence_id"})
    df = pd.merge(df, gt, on="sequence_id", how="left")
    return (df["sequence_id"].values,
            df["true_taxon_at_rank"].values,
            df["pred_label"].values,
            (df["true_status_at_rank"]=="NOVEL").astype(np.int8).values,
            df["is_novel"].astype(np.int8).values,
            df["z_score"].values)

def load_dnaberts(rank, seed):
    return _load_embedding_baseline("phase17_dnaberts_baseline", rank, seed)

def load_nt(rank, seed):
    return _load_embedding_baseline("phase18_nt_baseline", rank, seed)

LOADER = {
    "MultinomialNB": load_multinomialnb,
    "MSP-full": load_msp,
    "VSEARCH": load_vsearch,
    "Kraken2": load_kraken2,
    "DNABERT-S": load_dnaberts,
    "Nucleotide Transformer": load_nt,
    "HARBOUR": load_harbour,
}


# ── worker: process one (rank, seed) pair ──────────────────────────────────────
def process_rank_seed(rank, seed):
    """Return (mean_rows, bootstrap_rows) for one rank×seed."""
    # 1. Load all models
    raw = {}
    for model in MODELS:
        sid, ytc, ypc, ytb, ypb, ys = LOADER[model](rank, seed)
        raw[model] = {"sid": sid, "ytc": ytc, "ypc": ypc,
                      "ytb": ytb, "ypb": ypb, "ys": ys}

    # 2. Intersect sequence_ids
    common = set(raw[MODELS[0]]["sid"])
    for m in MODELS[1:]:
        common &= set(raw[m]["sid"])
    common = sorted(common)                       # deterministic order
    id2pos = {s: i for i, s in enumerate(common)}
    n = len(common)

    # 3. Reindex every model into aligned numpy arrays and PRE-ENCODE STRINGS to INTS
    aligned = {}
    for m in MODELS:
        r = raw[m]
        # Build lookup from original sid → position
        orig_map = {s: i for i, s in enumerate(r["sid"])}
        orig_idx = np.array([orig_map[s] for s in common])
        
        ytb_al = r["ytb"][orig_idx]
        ypb_al = r["ypb"][orig_idx]
        ytc_al = r["ytc"][orig_idx]
        ypc_al = r["ypc"][orig_idx]
        
        # Pre-encode strings for MCC and F1
        yt_mc = np.where(ytb_al == 1, "NOVEL", ytc_al.astype(str))
        yp_mc = np.where(ypb_al == 1, "NOVEL", ypc_al.astype(str))
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Fit on all possible labels for this model
        le.fit(np.concatenate([yt_mc, yp_mc]))
        ytc_enc = le.transform(yt_mc)
        ypc_enc = le.transform(yp_mc)
        
        aligned[m] = {
            "ytc": ytc_enc,
            "ypc": ypc_enc,
            "ytb": ytb_al,
            "ypb": ypb_al,
            "ys":  r["ys"][orig_idx],
        }

    # 4. Point-estimate metrics for this seed
    seed_metrics = {}
    for m in MODELS:
        a = aligned[m]
        seed_metrics[m] = calc_metrics(a["ytb"], a["ypb"], a["ys"],
                                       a["ytc"], a["ypc"])

    # 5. Paired bootstrap
    rng = np.random.RandomState(RNG_SEED + seed + hash(rank) % 10000)
    boot = {m: {k: [] for k in seed_metrics[MODELS[0]]} for m in MODELS}

    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        for m in MODELS:
            a = aligned[m]
            met = calc_metrics(a["ytb"][idx], a["ypb"][idx], a["ys"][idx],
                               a["ytc"][idx], a["ypc"][idx])
            for k, v in met.items():
                boot[m][k].append(v)

    boot_rows = []
    for m in MODELS:
        row = {"Model": m, "Rank": rank.capitalize(), "Seed": seed}
        for k, vals in boot[m].items():
            arr = np.array(vals)
            row[f"{k}_mean"] = np.nanmean(arr)
            row[f"{k}_lo"]   = np.nanpercentile(arr, 2.5)
            row[f"{k}_hi"]   = np.nanpercentile(arr, 97.5)
        boot_rows.append(row)

    mean_row_list = []
    for m in MODELS:
        r = {"Model": m, "Rank": rank.capitalize(), "Seed": seed}
        r.update(seed_metrics[m])
        mean_row_list.append(r)

    return mean_row_list, boot_rows, n


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"Phase 21 consolidation — {len(MODELS)} models × {len(RANKS)} ranks × {len(SEEDS)} seeds")
    print(f"Bootstrap B={B}, RNG seed={RNG_SEED}\n")

    all_mean_rows = []
    all_boot_rows = []

    for rank in RANKS:
        for seed in SEEDS:
            print(f"  [{rank}/seed_{seed}] ...", end=" ", flush=True)
            mr, br, n = process_rank_seed(rank, seed)
            all_mean_rows.extend(mr)
            all_boot_rows.extend(br)
            print(f"done  (n={n})")

    # ── Table 1: Baseline comparison (mean ± std across seeds) ─────────────
    df_mean = pd.DataFrame(all_mean_rows)
    metric_cols = ["H_score","AUROC","MCC","Novel_F1","Closed_set_F1","A_known","R_novel"]
    agg = df_mean.groupby(["Model","Rank"])[metric_cols].agg(["mean","std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    # Sort nicely
    rank_order = {r.capitalize(): i for i, r in enumerate(RANKS)}
    model_order = {m: i for i, m in enumerate(MODELS)}
    agg["_ro"] = agg["Rank"].map(rank_order)
    agg["_mo"] = agg["Model"].map(model_order)
    agg = agg.sort_values(["_ro","_mo"]).drop(columns=["_ro","_mo"])

    # Format mean ± std
    formatted_rows = []
    for _, row in agg.iterrows():
        fr = {"Model": row["Model"], "Rank": row["Rank"]}
        for mc in metric_cols:
            fr[mc] = f"{row[f'{mc}_mean']:.4f} ± {row[f'{mc}_std']:.4f}"
        formatted_rows.append(fr)
    df_table1 = pd.DataFrame(formatted_rows)
    df_table1.to_csv(f"{OUT_DIR}/phase21_baseline_table.csv", index=False)
    print(f"\n- Saved baseline table ({len(df_table1)} rows)")

    # ── Table 2: Bootstrap CIs (pooled across seeds) ───────────────────────
    df_boot = pd.DataFrame(all_boot_rows)
    # Aggregate bootstrap across seeds: pool the means
    boot_agg_rows = []
    for rank in RANKS:
        for model in MODELS:
            sub = df_boot[(df_boot["Model"]==model)&(df_boot["Rank"]==rank.capitalize())]
            row = {"Model": model, "Rank": rank.capitalize()}
            for mc in metric_cols:
                means = sub[f"{mc}_mean"].values
                los   = sub[f"{mc}_lo"].values
                his   = sub[f"{mc}_hi"].values
                # Grand mean and pooled CI
                gm = np.nanmean(means)
                gl = np.nanmean(los)
                gh = np.nanmean(his)
                row[mc] = f"{gm:.3f} [{gl:.3f}–{gh:.3f}]"
            boot_agg_rows.append(row)
    df_ci = pd.DataFrame(boot_agg_rows)
    df_ci.to_csv(f"{OUT_DIR}/phase21_bootstrap_ci.csv", index=False)
    print(f"- Saved bootstrap CIs ({len(df_ci)} rows)")

    # ── Table 3: Copy ablation table ───────────────────────────────────────
    abl_src = f"{REVAMP}/phase14_ablation/phase14_ablation_summary.csv"
    if os.path.exists(abl_src):
        import shutil
        shutil.copy2(abl_src, f"{OUT_DIR}/phase21_ablation_table.csv")
        print(f"- Copied ablation table")
    else:
        print(f"WARN Ablation table not found at {abl_src}")

    # ── Raw bootstrap data (for reproducibility) ──────────────────────────
    df_boot.to_csv(f"{OUT_DIR}/phase21_bootstrap_raw.csv", index=False)
    df_mean.to_csv(f"{OUT_DIR}/phase21_seed_level_metrics.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n- All done in {elapsed:.1f}s")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
