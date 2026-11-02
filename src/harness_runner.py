"""
harness_runner.py
─────────────────────────────────────────────────────────────────────────────
Robust entry point for lm-evaluation-harness that patches known runtime issues
in newer transformers, datasets (>= 3.0), and huggingface_hub environments:
  1. Translates load_in_8bit / load_in_4bit into BitsAndBytesConfig for transformers >= 4.46
  2. Resolves HfUriError on bare dataset names (gsm8k, hellaswag, winogrande, arc_easy) in huggingface_hub
  3. Bypasses strict repo_id mismatch in datasets.builder for patched/relocated datasets
  4. Provides QuantizeConfig fallback for optimum GPTQ quantizer

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import sys
import torch
import transformers
from urllib.parse import unquote
from transformers import BitsAndBytesConfig

# 1. Compatibility Patch: Convert legacy BnB kwargs to BitsAndBytesConfig
orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    load_8 = kwargs.pop("load_in_8bit", None)
    load_4 = kwargs.pop("load_in_4bit", None)
    
    # Handle string boolean values from CLI parsing
    if str(load_8).lower() in ["true", "1"]:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif str(load_4).lower() in ["true", "1"]:
        quant_type = kwargs.pop("bnb_4bit_quant_type", "nf4")
        kwargs.pop("bnb_4bit_compute_dtype", None)
        kwargs.pop("bnb_4bit_use_double_quant", None)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quant_type),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return orig_from_pretrained(*args, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained

# 2. Compatibility Patch: HfUriError & bare repository names across all HuggingFace datasets
try:
    import huggingface_hub.utils._hf_uris as hf_uris

    # Patch HfUri.__post_init__ to permit bare IDs (count('/') in [0, 1])
    orig_post_init = hf_uris.HfUri.__post_init__

    def patched_post_init(self):
        uri = self._raw or ""
        if not self.id:
            raise hf_uris.HfUriError(uri=uri, msg="Missing repository id.")
        if self.revision is not None and not self.revision:
            raise hf_uris.HfUriError(uri=uri, msg="Revision must not be an empty string.")

    hf_uris.HfUri.__post_init__ = patched_post_init

    # Patch _parse_repo_body to parse bare dataset names without raising HfUriError
    orig_parse_repo_body = hf_uris._parse_repo_body

    def robust_parse_repo_body(location: str, type_, *, raw: str = ""):
        location = location.strip("/")
        if not location:
            raise hf_uris.HfUriError(uri=raw, msg="Missing repository id.")

        at_idx = location.find("@")
        if at_idx == -1:
            parts = location.split("/", 1)
            if len(parts) == 1:
                repo_id = parts[0]
                path_in_repo = ""
            else:
                subparts = parts[1].split("/", 1)
                if len(subparts) == 1:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    path_in_repo = ""
                else:
                    repo_id = f"{parts[0]}/{subparts[0]}"
                    path_in_repo = subparts[1]
            return hf_uris.HfUri(type=type_, id=repo_id, revision=None, path_in_repo=path_in_repo, _raw=raw)
        else:
            repo_id = location[:at_idx]
            rev_and_path = location[at_idx + 1:]
            slash_idx = rev_and_path.find("/")
            if slash_idx == -1:
                revision = rev_and_path
                path_in_repo = ""
            else:
                revision = rev_and_path[:slash_idx]
                path_in_repo = rev_and_path[slash_idx + 1:]
            return hf_uris.HfUri(type=type_, id=repo_id, revision=unquote(revision), path_in_repo=path_in_repo, _raw=raw)

    hf_uris._parse_repo_body = robust_parse_repo_body
except Exception:
    pass

# 3. Compatibility Patch: datasets.builder.DatasetBuilder data_files check
try:
    import datasets.builder as ds_builder
    orig_builder_init = ds_builder.DatasetBuilder.__init__

    def patched_builder_init(self, *args, **kwargs):
        try:
            orig_builder_init(self, *args, **kwargs)
        except ValueError as e:
            if "Data files don't belong to" in str(e):
                orig_repo_id = kwargs.get("repo_id")
                kwargs["repo_id"] = None
                orig_builder_init(self, *args, **kwargs)
                self.repo_id = orig_repo_id
            else:
                raise e

    ds_builder.DatasetBuilder.__init__ = patched_builder_init
except Exception:
    pass

# 4. Compatibility Patch: Optimum GPTQ QuantizeConfig undefined NameError
try:
    import optimum.gptq.quantizer as gptq_quantizer
    if not hasattr(gptq_quantizer, "QuantizeConfig"):
        class FallbackQuantizeConfig:
            def __init__(self, *args, **kwargs):
                pass
        gptq_quantizer.QuantizeConfig = FallbackQuantizeConfig
except Exception:
    pass

# 5. Compatibility Patch: Suppress deprecated trust_remote_code warning and redundant HTTP lookups
try:
    import datasets
    import datasets.load as ds_load
    orig_load_dataset = ds_load.load_dataset

    def patched_load_dataset(*args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        return orig_load_dataset(*args, **kwargs)

    ds_load.load_dataset = patched_load_dataset
    if hasattr(datasets, "load_dataset"):
        datasets.load_dataset = patched_load_dataset
except Exception:
    pass

# Launch lm-eval CLI
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
