import os
import shutil
import glob

def main():
    target_dir = "E:/HTRP/master Q1 details"
    brain_dir = "C:/Users/chira/.gemini/antigravity-ide/brain/908df2a9-6be5-4838-a0b4-791f922f1917"
    htrp_dir = "E:/HTRP"
    
    # 1. Clear target directory
    if os.path.exists(target_dir):
        for f in glob.glob(f"{target_dir}/*"):
            if os.path.isfile(f):
                os.remove(f)
    else:
        os.makedirs(target_dir)
        
    print(f"Cleared {target_dir}")
    
    # 2. Collect files from HTRP dir (JSONs, CSVs generated during Phase 1 & 2)
    files_to_copy = [
        "phase2_acceptance_report.json",
        "phase2_acceptance_report.csv",
        "dataset_manifest.json",
        "leakage_report.csv",
        "baseline_metrics_16S.json",
        "benchmark_hierarchical.json",
        "benchmark_hierarchical_tuned.json",
        "benchmark_infonce.json",
        "benchmark_supcon.json",
        "train_metrics_infonce_16S.json",
        "train_metrics_supcon_16S.json"
    ]
    
    for f in files_to_copy:
        src = os.path.join(htrp_dir, f)
        dst = os.path.join(target_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {f}")
            
    # 3. Collect MD artifacts from brain directory
    brain_mds = glob.glob(f"{brain_dir}/*.md")
    for src in brain_mds:
        fname = os.path.basename(src)
        dst = os.path.join(target_dir, fname)
        shutil.copy2(src, dst)
        print(f"Copied artifact: {fname}")
        
    # 4. Consolidate MDs per phase
    # The user requested: "sonsolidate all of the outputs of a phase into md"
    # Phase 2 MDs:
    phase2_md = os.path.join(target_dir, "Phase_2_Consolidated_Outputs.md")
    with open(phase2_md, "w", encoding="utf-8") as out:
        out.write("# Phase 2 Consolidated Outputs\\n\\n")
        out.write("This document consolidates all Phase 2 analysis reports.\\n\\n")
        
        for md_file in ["leakage_report_summary.md", "walkthrough.md", "walkthrough_phase2a_1.md"]:
            path = os.path.join(target_dir, md_file)
            if os.path.exists(path):
                out.write(f"## {md_file}\\n\\n")
                with open(path, "r", encoding="utf-8") as infile:
                    out.write(infile.read())
                out.write("\\n\\n---\\n\\n")
                
    print("Consolidated Phase 2 MDs.")

if __name__ == "__main__":
    main()
