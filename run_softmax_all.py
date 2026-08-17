"""
Softmax-CNN Baseline: Parallel Orchestrator
Runs training + evaluation for all 20 rank×seed configurations using multiprocessing.
"""
import os
import sys
import json
import time
import csv
import concurrent.futures
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = "E:/HTRP"
RANKS = ["class", "order", "family", "genus"]
SEEDS = [42, 43, 44, 45, 46]
MAX_WORKERS = 10


def run_single_config(rank, seed):
    """Train and evaluate a single config. Returns summary dict."""
    try:
        from train_softmax_baseline import train_single_config
        from eval_softmax_baseline import eval_single_config

        print(f"\n{'='*50}")
        print(f"STARTING: {rank}/seed_{seed}")
        print(f"{'='*50}")

        t0 = time.time()

        # Phase 1: Train
        train_result = train_single_config(rank, seed, BASE_DIR, epochs=3, batch_size=64, lr=1e-3)

        # Phase 2: Evaluate
        eval_result = eval_single_config(rank, seed, BASE_DIR)

        total_time = time.time() - t0
        eval_result["total_time_s"] = total_time
        eval_result["train_time_s"] = train_result["train_time_s"]
        eval_result["status"] = "OK"

        print(f"\nCOMPLETED: {rank}/seed_{seed} in {total_time:.1f}s")
        return eval_result

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\nFAILED: {rank}/seed_{seed}: {error_msg}")
        return {
            "rank": rank,
            "seed": seed,
            "status": "FAILED",
            "error": str(e),
            "traceback": error_msg
        }


def main():
    print("=" * 70)
    print("SOFTMAX-CNN BASELINE: PARALLEL ORCHESTRATOR")
    print(f"Configs: {len(RANKS)} ranks × {len(SEEDS)} seeds = {len(RANKS)*len(SEEDS)} total")
    print(f"Workers: {MAX_WORKERS}")
    print("=" * 70)

    configs = [(rank, seed) for rank in RANKS for seed in SEEDS]
    results = []

    t_start = time.time()

    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_config = {
            executor.submit(run_single_config, rank, seed): (rank, seed)
            for rank, seed in configs
        }

        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
                ok_count = sum(1 for r in results if r.get("status") == "OK")
                fail_count = sum(1 for r in results if r.get("status") == "FAILED")
                print(f"\n[Progress] {ok_count} OK, {fail_count} FAILED, {len(configs) - len(results)} remaining")
            except Exception as e:
                print(f"\n[ERROR] {config}: {e}")
                results.append({"rank": config[0], "seed": config[1], "status": "FAILED", "error": str(e)})

    total_time = time.time() - t_start

    # ---- Merge results ----
    output_dir = f"{BASE_DIR}/results/softmax_baseline"
    os.makedirs(output_dir, exist_ok=True)

    # Filter successful results
    ok_results = [r for r in results if r.get("status") == "OK"]
    failed_results = [r for r in results if r.get("status") == "FAILED"]

    if ok_results:
        # Write summary CSV
        summary_file = f"{output_dir}/softmax_baseline_summary.csv"
        fieldnames = [k for k in ok_results[0].keys() if k not in ("status", "traceback", "error")]
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted(ok_results, key=lambda x: (x["rank"], x["seed"])):
                writer.writerow({k: r.get(k, "") for k in fieldnames})

        # Compute per-rank aggregate stats
        import numpy as np
        agg_rows = []
        metrics_to_agg = ["A_known", "R_novel", "H_Score", "AUROC", "AUPRC",
                          "accuracy", "macro_precision", "macro_recall", "macro_f1",
                          "weighted_f1", "balanced_accuracy", "MCC",
                          "FK_rate", "FN_rate", "wrong_taxon_rate"]

        for rank in RANKS:
            rank_results = [r for r in ok_results if r["rank"] == rank]
            if not rank_results:
                continue
            agg = {"rank": rank, "n_seeds": len(rank_results)}
            for metric in metrics_to_agg:
                vals = [r[metric] for r in rank_results if not (isinstance(r[metric], float) and np.isnan(r[metric]))]
                if vals:
                    agg[f"{metric}_mean"] = np.mean(vals)
                    agg[f"{metric}_std"] = np.std(vals)
                else:
                    agg[f"{metric}_mean"] = float('nan')
                    agg[f"{metric}_std"] = float('nan')
            agg_rows.append(agg)

        agg_file = f"{output_dir}/softmax_baseline_aggregate.csv"
        if agg_rows:
            with open(agg_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=agg_rows[0].keys())
                writer.writeheader()
                writer.writerows(agg_rows)

        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Successful: {len(ok_results)}/{len(configs)}")
        print(f"Failed: {len(failed_results)}/{len(configs)}")
        print(f"Total wall-clock: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"\nPer-rank aggregate (mean ± std across seeds):")
        print(f"{'Rank':<10} {'H_Score':<20} {'AUROC':<20} {'Accuracy':<20} {'MCC':<20}")
        for agg in agg_rows:
            h = f"{agg.get('H_Score_mean', 0):.4f}±{agg.get('H_Score_std', 0):.4f}"
            a = f"{agg.get('AUROC_mean', 0):.4f}±{agg.get('AUROC_std', 0):.4f}"
            acc = f"{agg.get('accuracy_mean', 0):.4f}±{agg.get('accuracy_std', 0):.4f}"
            mcc = f"{agg.get('MCC_mean', 0):.4f}±{agg.get('MCC_std', 0):.4f}"
            print(f"{agg['rank']:<10} {h:<20} {a:<20} {acc:<20} {mcc:<20}")

        print(f"\nOutputs saved to:")
        print(f"  Summary: {summary_file}")
        print(f"  Aggregate: {agg_file}")
    else:
        print("\nALL CONFIGS FAILED!")

    if failed_results:
        print(f"\nFailed configs:")
        for r in failed_results:
            print(f"  {r['rank']}/seed_{r['seed']}: {r.get('error', 'unknown')}")

    # Save full run report
    report = {
        "total_configs": len(configs),
        "successful": len(ok_results),
        "failed": len(failed_results),
        "total_wall_clock_s": total_time,
        "max_workers": MAX_WORKERS,
        "failed_details": [{"rank": r["rank"], "seed": r["seed"], "error": r.get("error")} for r in failed_results]
    }
    with open(f"{output_dir}/run_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
