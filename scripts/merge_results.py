#!/usr/bin/env python3
"""
merge_results.py
─────────────────────────────────────────────────────────────────────────────
Merges evaluation results from separate model runs (e.g., 3 separate Kaggle
notebooks: Qwen, Llama, TinyLlama) into a unified master results CSV.
Automatically eliminates duplicate rows, keeps successful entries, and
re-runs statistical analysis across the full 3-model suite.

Usage:
    python scripts/merge_results.py results/*.csv --output results/results_master.csv --analyze

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

def merge_csvs(input_paths, output_path: str = "results/results_master.csv") -> pd.DataFrame:
    dfs = []
    for p in input_paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".csv":
            try:
                df = pd.read_csv(path)
                if not df.empty and "model" in df.columns:
                    dfs.append(df)
                    print(f"[merge] Loaded {len(df)} rows from {path}")
            except Exception as e:
                print(f"[merge] Warning: Could not read {path}: {e}")
                
    if not dfs:
        print("[merge] Error: No valid results CSV files found.")
        sys.exit(1)
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate: if multiple entries exist for (model, variant), prefer status=='SUCCESS'
    if "status" in combined.columns:
        combined["is_success"] = (combined["status"] == "SUCCESS").astype(int)
        combined = combined.sort_values(by=["is_success", "timestamp"], ascending=[True, True])
        combined = combined.drop_duplicates(subset=["model", "variant"], keep="last")
        combined = combined.drop(columns=["is_success"])
    else:
        combined = combined.drop_duplicates(subset=["model", "variant"], keep="last")
        
    combined = combined.sort_values(by=["model", "bits"], ascending=[True, False])
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\n[merge] ✅ Successfully merged {len(combined)} unique configurations into: {output_path}")
    return combined

def main():
    parser = argparse.ArgumentParser(description="Merge results from multiple Kaggle notebook runs")
    parser.add_argument("inputs", nargs="+", help="Input CSV files to merge (e.g. results/results_master.csv other_results.csv)")
    parser.add_argument("--output", default="results/results_master.csv", help="Target output CSV path")
    parser.add_argument("--analyze", action="store_true", help="Automatically trigger statistical analysis on the merged results")
    args = parser.parse_args()
    
    merged_df = merge_csvs(args.inputs, args.output)
    
    if args.analyze:
        import subprocess
        print("\n[merge] Triggering 10,000-iteration Paired Bootstrap Statistical Analysis...")
        subprocess.run([sys.executable, "run_analysis.py", "--csv", args.output, "--bootstrap", "10000"])

if __name__ == "__main__":
    main()
