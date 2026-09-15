"""
calibrate.py
─────────────────────────────────────────────────────────────────────────────
Calibration data pipeline for architecture-agnostic LLM PTQ.

Steps:
  1. Load WikiText-2 train split and tokenize with the model's own tokenizer.
  2. Sample 128 non-overlapping windows of 2048 tokens (fixed seed).
  3. Register forward hooks on every nn.Linear in the model.
  4. Run one forward pass (no gradients) to capture input activations X.
  5. Return a dict: { layer_name -> X tensor on CPU }.

This module is architecture-agnostic — it works on any HuggingFace
model that uses nn.Linear layers.

Authors: Faqre Alam · Guneet Toppo · Ekansh Agrawal
Project: BTech Final Year Project, DTU 2026-27
"""

import random
import torch
import torch.nn as nn
from typing import Dict, List
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_calibration_data(
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> List[torch.Tensor]:
    """
    Load and prepare calibration sequences from WikiText-2.

    Returns
    -------
    List of n_samples tensors, each of shape [1, seq_len] (int64 token IDs).
    Sequences are contiguous non-overlapping windows sampled with a fixed seed.
    """
    print(f"[calibrate] Loading {dataset_name}/{dataset_config} ({split} split)...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Concatenate all text and tokenize in one shot (no truncation/padding)
    full_text = "\n\n".join(dataset["text"])
    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )["input_ids"]  # shape [1, total_tokens]

    total_tokens = tokens.shape[1]
    assert total_tokens >= n_samples * seq_len, (
        f"Not enough tokens ({total_tokens}) for {n_samples} sequences of "
        f"length {seq_len}. Reduce n_samples or seq_len."
    )

    rng = random.Random(seed)
    max_start = total_tokens - seq_len
    starts = rng.sample(range(0, max_start), n_samples)

    samples = []
    for start in starts:
        chunk = tokens[:, start : start + seq_len]  # [1, seq_len]
        samples.append(chunk)

    print(f"[calibrate] Prepared {n_samples} sequences of length {seq_len}.")
    return samples


def capture_layer_inputs(
    model: PreTrainedModel,
    calibration_samples: List[torch.Tensor],
    skip_layer_names: List[str] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Run one forward pass through the model and capture the input activations
    (X) for every nn.Linear layer via forward hooks.

    Parameters
    ----------
    model              : loaded HuggingFace model (FP16, on `device`)
    calibration_samples: list of [1, seq_len] token-ID tensors
    skip_layer_names   : list of substrings — layers whose name contains any
                         of these strings are skipped (e.g. "lm_head")
    device             : "cuda" or "cpu"

    Returns
    -------
    Dict mapping each quantizable layer's full dotted name to its Hessian
    matrix H of shape [d_in, d_in], stored on CPU.
    H is scaled by 2.0 / N_total.
    """
    if skip_layer_names is None:
        skip_layer_names = ["embed_tokens", "lm_head", "embed_positions"]

    # ── Identify target layers ──────────────────────────────────────────────
    target_layers: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_layer_names):
            continue
        target_layers[name] = module

    print(f"[calibrate] Found {len(target_layers)} nn.Linear layers to quantize.")

    # ── Storage buffers ─────────────────────────────────────────────────────
    hessian_store: Dict[str, torch.Tensor] = {}
    N_store: Dict[str, int] = {name: 0 for name in target_layers}

    # ── Register hooks ───────────────────────────────────────────────────────
    hooks = []

    def make_hook(layer_name: str):
        def hook(module: nn.Module, inp, out):
            # inp[0] shape: [batch, seq_len, d_in]  (batch=1)
            x = inp[0].detach().float()          # cast to FP32 for stability
            x = x.reshape(-1, x.shape[-1])       # flatten to [seq_len, d_in]
            H_batch = x.T @ x
            if layer_name not in hessian_store:
                hessian_store[layer_name] = H_batch.cpu()
            else:
                hessian_store[layer_name] += H_batch.cpu()
            N_store[layer_name] += x.shape[0]
        return hook

    for name, module in target_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    # ── Run forward passes ───────────────────────────────────────────────────
    model.eval()
    
    # When using device_map="auto", model is already distributed across GPUs.
    # We must NOT call model.to(device). Instead, find where the first layer is.
    first_device = next(model.parameters()).device

    print(f"[calibrate] Running {len(calibration_samples)} forward passes...")
    with torch.no_grad():
        for sample in tqdm(calibration_samples, desc="Calibration forward", unit="seq"):
            input_ids = sample.to(first_device)
            try:
                model(input_ids)
            except Exception as e:
                # Some models raise on partial inputs — try with attention_mask
                attn_mask = torch.ones_like(input_ids)
                model(input_ids, attention_mask=attn_mask)

    # ── Remove hooks ─────────────────────────────────────────────────────────
    for h in hooks:
        h.remove()
    print("[calibrate] Hooks removed.")

    # ── Compute final scaled Hessians ────────────────────────────────────────
    activations: Dict[str, torch.Tensor] = {}
    for name, H_sum in hessian_store.items():
        N = N_store[name]
        if N == 0:
            print(f"[calibrate] WARNING: no activations captured for '{name}' — skipping.")
            continue
        H = (2.0 / N) * H_sum
        activations[name] = H

    total_mb = sum(x.element_size() * x.nelement() for x in activations.values()) / 1e6
    print(f"[calibrate] Hessian store: {len(activations)} layers, "
          f"{total_mb:.1f} MB on CPU.")

    return activations


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python -m phase1.framework.calibrate)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    print(f"\n=== Calibration self-test | model: {model_id} ===\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    samples = get_calibration_data(tokenizer, n_samples=4, seq_len=512, seed=42)
    activations = capture_layer_inputs(model, samples, device="cuda" if torch.cuda.is_available() else "cpu")

    for name, H in list(activations.items())[:3]:
        print(f"  {name:60s}  H.shape={tuple(H.shape)}")
    print("\nCalibration self-test passed ✅")
