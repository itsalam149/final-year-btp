"""
model_loader.py
─────────────────────────────────────────────────────────────────────────────
Universal model loading adapter for official and community quantized LLMs.
Supports:
  1. Standard FP16 baselines
  2. BitsAndBytes 8-bit (Int8) & 4-bit (NF4)
  3. Pre-quantized GPTQ checkpoints (official & community)
  4. Pre-quantized AWQ checkpoints (official & community)

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import os
import sys
import torch
from typing import Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

def detect_device() -> str:
    """Auto-detect optimal compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_lm_eval_model_args(variant_cfg: Dict[str, Any], device: str = "auto") -> str:
    """
    Construct the standardized model_args string for lm-evaluation-harness.
    
    lm-eval accepts:
      --model hf --model_args pretrained=<id>,dtype=float16,...
    """
    hf_id = variant_cfg["hf_id"]
    loader = variant_cfg.get("loader", "transformers")
    method = variant_cfg.get("method", "fp16")
    
    args_list = [f"pretrained={hf_id}"]
    
    if loader == "transformers":
        args_list.append("dtype=float16")
    elif loader == "bitsandbytes":
        if method == "bnb_int8":
            args_list.append("load_in_8bit=True")
        elif method == "bnb_nf4":
            args_list.append("load_in_4bit=True")
            args_list.append("bnb_4bit_quant_type=nf4")
            args_list.append("bnb_4bit_compute_dtype=float16")
    elif loader in ["gptq", "awq"]:
        # Pre-quantized GPTQ/AWQ checkpoints contain quantization_config in config.json.
        # Transformers AutoModelForCausalLM automatically handles them natively.
        pass
        
    return ",".join(args_list)

def load_hf_model(
    hf_id: str,
    loader: str = "transformers",
    method: str = "fp16",
    device: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Direct Python loader for interactive testing or verification.
    """
    if device is None:
        device = detect_device()
        
    print(f"[loader] Loading: {hf_id} (loader={loader}, method={method}, target_device={device})")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if loader == "bitsandbytes":
        from transformers import BitsAndBytesConfig
        if method == "bnb_int8":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif method == "bnb_nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            raise ValueError(f"Unknown bnb method: {method}")
            
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb_config,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
    elif loader == "gptq":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
    elif loader == "awq":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=True
        )
    else:
        # Standard FP16
        if device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to("mps")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=torch.float16,
                device_map="auto" if device != "cpu" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
    model.eval()
    print(f"[loader] Successfully loaded {hf_id} on {device}.")
    return model, tokenizer

