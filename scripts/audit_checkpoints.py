#!/usr/bin/env python3
"""
audit_checkpoints.py
─────────────────────────────────────────────────────────────────────────────
Phase B+ Checkpoint Provenance Audit Utility (BTP GP159).
Inspects all 11 model checkpoints specified in configs/checkpoints.yaml:
  - Queries Hugging Face Hub for exact commit SHA, file sizes, and licenses.
  - Inspects config.json and quantize_config.json for quantization parameters:
    (bits, group_size, sym, desc_act, calibration dataset).
  - Updates configs/checkpoints.yaml with pinned commit hashes for strict reproducibility.

Usage:
    python scripts/audit_checkpoints.py
    python scripts/audit_checkpoints.py --update-yaml

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_checkpoints_yaml(path: str = "configs/checkpoints.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def audit_checkpoint(repo_id: str, token: str = None) -> Dict[str, Any]:
    """
    Queries Hugging Face model metadata and inspects configuration files.
    """
    if not HF_AVAILABLE:
        return {"status": "ERROR", "message": "huggingface_hub package not available"}

    api = HfApi(token=token)
    audit_data = {
        "repo_id": repo_id,
        "status": "UNKNOWN",
        "commit_sha": None,
        "is_gated": False,
        "license": None,
        "size_mb": None,
        "quant_config": {},
    }

    try:
        model_info = api.model_info(repo_id, token=token, files_metadata=True)
        audit_data["status"] = "VERIFIED"
        audit_data["commit_sha"] = model_info.sha
        audit_data["license"] = getattr(model_info, "cardData", {}).get("license", None)

        # Estimate disk footprint from sibling files
        if model_info.siblings:
            total_bytes = sum(getattr(s, "size", 0) or 0 for s in model_info.siblings)
            audit_data["size_mb"] = round(total_bytes / (1024 * 1024), 2)

        # Download config.json if accessible
        try:
            config_file = hf_hub_download(repo_id, "config.json", token=token)
            with open(config_file) as f:
                cfg = json.load(f)

            # Check for quantization config inside config.json or separate file
            if "quantization_config" in cfg:
                audit_data["quant_config"] = cfg["quantization_config"]
        except Exception:
            pass

        # Check for quantize_config.json (standard for GPTQ/AWQ)
        try:
            q_file = hf_hub_download(repo_id, "quantize_config.json", token=token)
            with open(q_file) as f:
                audit_data["quant_config"].update(json.load(f))
        except Exception:
            pass

    except GatedRepoError:
        audit_data["status"] = "GATED_ACCESS_REQUIRED"
        audit_data["is_gated"] = True
    except RepositoryNotFoundError:
        audit_data["status"] = "REPO_NOT_FOUND"
    except Exception as e:
        audit_data["status"] = "NETWORK_OR_API_ERROR"
        audit_data["error"] = str(e)

    return audit_data


def main():
    parser = argparse.ArgumentParser(description="Audit Hugging Face checkpoints for Phase B+")
    parser.add_argument("--config", default="configs/checkpoints.yaml", help="Path to checkpoints YAML")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face access token")
    parser.add_argument("--update-yaml", action="store_true", help="Update YAML with retrieved commit SHAs")
    args = parser.parse_args()

    cfg = load_checkpoints_yaml(args.config)
    checkpoints = cfg.get("checkpoints", {})

    print("\n" + "═" * 78)
    print("  PHASE B+ CHECKPOINT PROVENANCE AUDIT (BTP GP159)")
    print("═" * 78)
    print(f"  Total Checkpoints to Audit: {len(checkpoints)}\n")

    verified_count = 0
    gated_count = 0
    failed_count = 0

    for name, c_info in checkpoints.items():
        repo_id = c_info.get("repo_id")
        print(f"[{name}] Auditing: {repo_id} ...")
        res = audit_checkpoint(repo_id, token=args.token)
        print(f"  Status    : {res['status']}")

        if res["commit_sha"]:
            print(f"  Commit SHA: {res['commit_sha']}")
            if args.update_yaml:
                c_info["commit_hash"] = res["commit_sha"]

        if res["size_mb"]:
            print(f"  Size (MB) : {res['size_mb']} MB")
            if args.update_yaml:
                c_info["size_on_disk_mb"] = res["size_mb"]

        if res["quant_config"]:
            print(f"  Quant Config: {res['quant_config']}")
            if "bits" in res["quant_config"]:
                c_info["precision_bits"] = res["quant_config"]["bits"]
            if "group_size" in res["quant_config"]:
                c_info["group_size"] = res["quant_config"]["group_size"]

        if res["status"] == "VERIFIED":
            verified_count += 1
        elif res["status"] == "GATED_ACCESS_REQUIRED":
            gated_count += 1
        else:
            failed_count += 1
        print("─" * 60)

    if args.update_yaml:
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n[audit] Updated {args.config} with pinned commit hashes.")

    print("\n" + "═" * 78)
    print(f"  AUDIT SUMMARY: {verified_count} Verified, {gated_count} Gated, {failed_count} Blocked/Error.")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()

