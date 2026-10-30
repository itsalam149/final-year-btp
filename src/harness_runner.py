"""
harness_runner.py
─────────────────────────────────────────────────────────────────────────────
Robust entry point for lm-evaluation-harness that patches known runtime issues
in newer transformers and huggingface_hub environments:
  1. Translates load_in_8bit / load_in_4bit into BitsAndBytesConfig for transformers >= 4.46
  2. Resolves HfUriError on bare dataset names like hellaswag in huggingface_hub >= 0.29
  3. Provides QuantizeConfig fallback for optimum GPTQ quantizer

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import sys
import torch
import transformers
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

# 2. Compatibility Patch: HfUriError for legacy bare dataset repo names (e.g. hellaswag)
try:
    import huggingface_hub.utils._hf_uris as hf_uris
    orig_parse_repo_body = hf_uris._parse_repo_body

    def patched_parse_repo_body(location, type_, raw=None):
        try:
            return orig_parse_repo_body(location, type_, raw=raw)
        except Exception as e:
            if location == "hellaswag" or location.startswith("hellaswag@"):
                new_loc = location.replace("hellaswag", "Rowan/hellaswag", 1)
                new_raw = raw.replace("hellaswag", "Rowan/hellaswag", 1) if raw else None
                return orig_parse_repo_body(new_loc, type_, raw=new_raw)
            raise e

    hf_uris._parse_repo_body = patched_parse_repo_body
except Exception:
    pass

# 3. Compatibility Patch: Optimum GPTQ QuantizeConfig undefined NameError
try:
    import optimum.gptq.quantizer as gptq_quantizer
    if not hasattr(gptq_quantizer, "QuantizeConfig"):
        class FallbackQuantizeConfig:
            def __init__(self, *args, **kwargs):
                pass
        gptq_quantizer.QuantizeConfig = FallbackQuantizeConfig
except Exception:
    pass

# Launch lm-eval CLI
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
