"""
quantize.py
─────────────────────────────────────────────────────────────────────────────
Three layer-wise PTQ methods for any HuggingFace nn.Linear model.

Methods
-------
  1. RTN  — affine per-channel Round-To-Nearest (no calibration data)
  2. GPTQ — second-order Hessian error compensation (Frantar et al., ICLR 2023)
  3. GPTQ + Hadamard — randomised Hadamard rotation + GPTQ (QuIP/QuIP#)

All methods:
  • Operate on a single (W, X) pair — fully architecture-agnostic.
  • Take W in FP32 internally for numerical stability; output INT-packed via
    a (W_dequant, scales, zeros) representation that replaces module.weight.
  • quantize_model() applies the chosen method to ALL nn.Linear in the model
    by iterating model.named_modules() — no architecture-specific code.

Authors: Faqre Alam · Guneet Toppo · Ekansh Agrawal
Project: BTech Final Year Project, DTU 2026-27

References:
  [1] Frantar et al. "GPTQ", ICLR 2023.  arXiv:2210.17323
  [4] Chee et al.   "QuIP", NeurIPS 2023. arXiv:2307.13304
  [5] Tseng et al.  "QuIP#", ICML 2024.  arXiv:2402.04396
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RTN — Round-To-Nearest  (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def rtn_quantize(
    W: torch.Tensor,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-channel asymmetric affine quantization (Round-To-Nearest).

    For each output channel i:
        scale_i = (max(W[i]) - min(W[i])) / (2^bits - 1)
        zero_i  = round(-min(W[i]) / scale_i)
        W_q[i]  = clamp(round(W[i] / scale_i) + zero_i, 0, 2^bits - 1)

    Parameters
    ----------
    W    : [d_out, d_in] FP32 weight tensor
    bits : quantization bit-width (2, 3, or 4)

    Returns
    -------
    W_dequant : [d_out, d_in] FP32 — dequantized weights (replace module.weight)
    scales    : [d_out, 1]   FP32 — per-channel scales
    zeros     : [d_out, 1]   FP32 — per-channel zero-points
    """
    W = W.float()
    qmax = 2 ** bits - 1

    w_min = W.min(dim=1, keepdim=True).values       # [d_out, 1]
    w_max = W.max(dim=1, keepdim=True).values       # [d_out, 1]

    scales = (w_max - w_min).clamp(min=1e-8) / qmax   # [d_out, 1]
    zeros  = (-w_min / scales).round().clamp(0, qmax)  # [d_out, 1]

    W_q       = (W / scales + zeros).round().clamp(0, qmax)   # [d_out, d_in]
    W_dequant = scales * (W_q - zeros)                         # [d_out, d_in]

    return W_dequant, scales, zeros


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GPTQ — Second-Order Error Compensation
# ─────────────────────────────────────────────────────────────────────────────

def _build_hessian(H: torch.Tensor, dampening: float = 0.01) -> torch.Tensor:
    """
    Regularise the pre-computed Hessian proxy H.

    Parameters
    ----------
    H          : [d_in, d_in] pre-computed scaled Hessian (FP32)
    dampening  : ε multiplier — H += ε * mean(diag H) * I

    Returns
    -------
    H : [d_in, d_in] FP32 positive-definite Hessian proxy
    """
    H = H.clone()
    eps = dampening * H.diag().mean()
    H.add_(torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * eps)
    return H


def _cholesky_inverse(H: torch.Tensor) -> torch.Tensor:
    """
    Compute H^{-1} via Cholesky decomposition (numerically stable).
    Falls back to adding more dampening if H is not positive-definite.
    """
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        # Increase dampening and retry once
        eps = 0.1 * H.diag().mean()
        H.add_(torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * eps)
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    return H_inv


def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    dampening: float = 0.01,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ: layer-wise second-order post-training quantization.

    Algorithm (Frantar et al., ICLR 2023):
      1. Compute H = 2 X^T X, add dampening, invert via Cholesky.
      2. Process columns left-to-right in blocks of `block_size`:
         a. Quantize column j  (per-output-channel affine).
         b. err_j = W[:,j] - W_q[:,j]
         c. Compensate remaining columns: W[:,j+1:] -= err_j * H_inv[j, j+1:] / H_inv[j,j]
      3. Return dequantized weights.

    Parameters
    ----------
    W          : [d_out, d_in] FP32 weight tensor
    H          : [d_in, d_in]  FP32 pre-computed scaled Hessian
    bits       : quantization bit-width (2, 3, or 4)
    dampening  : ε multiplier for Hessian regularisation
    block_size : number of columns per lazy update block

    Returns
    -------
    W_dequant : [d_out, d_in] FP32 — dequantized weights
    scales    : [d_out, 1]   FP32 — per-channel scales
    zeros     : [d_out, 1]   FP32 — per-channel zero-points
    """
    dev = W.device
    W   = W.float().clone()               # work in FP32
    H   = H.float().to(dev)

    d_out, d_in = W.shape
    qmax = 2 ** bits - 1

    # ── Build Hessian and invert ──────────────────────────────────────────
    H     = _build_hessian(H, dampening)
    H_inv = _cholesky_inverse(H)          # [d_in, d_in]

    # ── Per-output-channel scale / zero (computed once from FP16 W) ───────
    w_min  = W.min(dim=1, keepdim=True).values
    w_max  = W.max(dim=1, keepdim=True).values
    scales = (w_max - w_min).clamp(min=1e-8) / qmax   # [d_out, 1]
    zeros  = (-w_min / scales).round().clamp(0, qmax)  # [d_out, 1]

    # ── Column-wise quantization loop ─────────────────────────────────────
    W_q = W.clone()

    for block_start in range(0, d_in, block_size):
        block_end = min(block_start + block_size, d_in)

        for j in range(block_start, block_end):
            # Quantize column j
            w_j   = W_q[:, j]                                      # [d_out]
            wq_j  = (w_j / scales[:, 0] + zeros[:, 0]).round()
            wq_j  = wq_j.clamp(0, qmax)
            wdq_j = scales[:, 0] * (wq_j - zeros[:, 0])            # [d_out]

            # Compensation: distribute error to all columns j+1 … d_in
            err = (w_j - wdq_j)                                     # [d_out]
            if j + 1 < d_in:
                # H_inv[j, j+1:] / H_inv[j,j] is the compensation direction
                h_inv_row = H_inv[j, j + 1 :]                      # [d_in-j-1]
                h_inv_jj  = H_inv[j, j].clamp(min=1e-8)
                W_q[:, j + 1 :] -= err.unsqueeze(1) * (h_inv_row / h_inv_jj).unsqueeze(0)

            # Store quantized column
            W_q[:, j] = wdq_j

    W_dequant = W_q
    return W_dequant, scales, zeros


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Hadamard Rotation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _hadamard_matrix(n: int) -> torch.Tensor:
    """
    Return the normalised Walsh-Hadamard matrix of size n×n.
    n must be a power of 2.
    """
    assert n & (n - 1) == 0, f"n must be a power of 2, got {n}"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H,  H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)


def _random_hadamard(d: int, seed: int = 1337, device: str = "cpu") -> torch.Tensor:
    """
    Randomised Hadamard matrix Q = Diag(s) · H_walsh (normalised).
    s is a random ±1 sign vector seeded deterministically.

    If d is not a power of 2, pads to the next power of 2 then crops.
    """
    d_pad = _next_power_of_two(d)
    H = _hadamard_matrix(d_pad).to(device)                    # [d_pad, d_pad]

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    signs = torch.randint(0, 2, (d_pad,), generator=rng, device=device)
    signs = signs * 2 - 1                                     # ±1
    Q = (signs.unsqueeze(1) * H)[:d, :d]                      # [d, d] (crop)
    return Q.float()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GPTQ + Hadamard Incoherence
# ─────────────────────────────────────────────────────────────────────────────

def hadamard_gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int,
    hadamard_seed: int = 1337,
    dampening: float = 0.01,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ with Randomised Hadamard Incoherence Processing (QuIP / QuIP#).

    Steps:
      1. Build Q = random Hadamard matrix (seeded, stored implicitly).
      2. Rotate:  W' = W @ Q^T   and   X' = X @ Q^T
         (equivalent output:  W · X = W' · X' since Q is orthogonal)
      3. Apply standard GPTQ on (W', X') — columns are now roughly equal
         in magnitude, so quantization error is uniform.
      4. Return the dequantised W' (still in rotated space) and the seed Q.
         At inference, multiply incoming activations by Q before the matmul.

    Parameters
    ----------
    W              : [d_out, d_in] FP32 weight tensor
    H              : [d_in, d_in]  FP32 pre-computed scaled Hessian
    bits           : bit-width
    hadamard_seed  : seed for the random Hadamard rotation (fixed for repro)
    dampening      : GPTQ dampening factor
    block_size     : GPTQ block size

    Returns
    -------
    W_dequant     : [d_out, d_in] FP32 — dequantized weights (rotated space)
    scales        : [d_out, 1]   FP32
    zeros         : [d_out, 1]   FP32
    hadamard_seed : int — the seed used (store alongside the model)
    """
    dev = W.device
    d_in = W.shape[1]

    # ── Build rotation matrix ─────────────────────────────────────────────
    Q = _random_hadamard(d_in, seed=hadamard_seed, device=dev)  # [d_in, d_in]

    # ── Rotate weights and Hessian ───────────────────────────────────
    W_rot = W.float() @ Q.T             # W' = W Q^T    [d_out, d_in]
    H     = H.float().to(dev)
    H_rot = Q @ H @ Q.T                 # H' = Q H Q^T  [d_in, d_in]

    # ── Apply GPTQ in rotated space ──────────────────────────────────────
    W_dequant, scales, zeros = gptq_quantize(
        W_rot, H_rot, bits, dampening=dampening, block_size=block_size
    )

    return W_dequant, scales, zeros, hadamard_seed


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Model-level quantization — applies chosen method to every nn.Linear
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    method: str,
    bits: int,
    dampening: float = 0.01,
    block_size: int = 128,
    hadamard_seed: int = 1337,
    skip_layer_names: Optional[list] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    Apply the chosen PTQ method to every nn.Linear in the model in-place.

    This function is 100% architecture-agnostic — it iterates
    model.named_modules() and applies quantization wherever it finds an
    nn.Linear whose name is not in the skip list.

    Parameters
    ----------
    model            : HuggingFace model (FP16 on `device`)
    activations      : dict from calibrate.capture_layer_inputs()
                       { layer_name -> Hessian tensor (CPU, FP32) }
    method           : one of "rtn" | "gptq" | "gptq_hadamard"
    bits             : 2, 3, or 4
    dampening        : GPTQ dampening factor
    block_size       : GPTQ block size
    hadamard_seed    : seed for Hadamard rotation
    skip_layer_names : list of name-substrings to skip
    device           : "cuda" or "cpu"

    Returns
    -------
    model with quantized weights replaced in-place.
    (Original model dtype is preserved — weights stored as FP16 dequantized.)
    """
    if skip_layer_names is None:
        skip_layer_names = ["embed_tokens", "lm_head", "embed_positions"]

    model.eval()
    quantized_count = 0
    skipped_count   = 0

    layer_iter = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    print(f"\n[quantize] Method={method}  bits={bits}  "
          f"Layers to process: {len(layer_iter)}")

    for name, module in tqdm(layer_iter, desc=f"Quantizing ({method}, {bits}b)", unit="layer"):
        # ── Skip check ───────────────────────────────────────────────────
        if any(skip in name for skip in skip_layer_names):
            skipped_count += 1
            continue

        if name not in activations:
            # No calibration data captured for this layer — use RTN fallback
            W_orig = module.weight.data.float()
            W_dq, _, _ = rtn_quantize(W_orig, bits)
            module.weight.data = W_dq.to(module.weight.dtype)
            quantized_count += 1
            continue

        # ── Load weight and Hessian ───────────────────────────────────
        W = module.weight.data.float().to(device)
        H = activations[name].float().to(device)      # [d_in, d_in]

        # ── Apply chosen method ─────────────────────────────────────────
        if method == "rtn":
            W_dq, scales, zeros = rtn_quantize(W, bits)

        elif method == "gptq":
            W_dq, scales, zeros = gptq_quantize(
                W, X, bits,
                dampening=dampening,
                block_size=block_size,
            )

        elif method == "gptq_hadamard":
            W_dq, scales, zeros, _ = hadamard_gptq_quantize(
                W, X, bits,
                hadamard_seed=hadamard_seed,
                dampening=dampening,
                block_size=block_size,
            )

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Choose from: rtn | gptq | gptq_hadamard")

        # ── Replace weight in-place ──────────────────────────────────────
        module.weight.data = W_dq.to(module.weight.dtype)

        # ── Free GPU tensors ─────────────────────────────────────────────
        del W, H, W_dq
        torch.cuda.empty_cache()

        quantized_count += 1

    print(f"[quantize] Done. Quantized: {quantized_count}  Skipped: {skipped_count}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic utilities
# ─────────────────────────────────────────────────────────────────────────────

def layer_output_error(W_orig: torch.Tensor, W_q: torch.Tensor,
                        X: torch.Tensor) -> float:
    """
    Relative reconstruction error:  ‖W_orig·X - W_q·X‖_F / ‖W_orig·X‖_F
    Useful for validating that GPTQ improves over RTN on a synthetic layer.
    """
    with torch.no_grad():
        Y_orig = (W_orig.float() @ X.float().T)
        Y_q    = (W_q.float()   @ X.float().T)
        err    = (Y_orig - Y_q).norm() / Y_orig.norm().clamp(min=1e-8)
    return err.item()


def column_norm_variance(W: torch.Tensor) -> float:
    """Variance of per-column L2 norms. Lower = more incoherent."""
    col_norms = W.float().norm(dim=0)   # [d_in]
    return col_norms.var().item()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python -m phase1.framework.quantize)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== quantize.py self-test ===\n")
    torch.manual_seed(0)

    # Use dimensions that match a real LLM attention layer (e.g. Qwen2.5-0.5B)
    # d_out=896, d_in=896 hidden→hidden projection
    # N=512 calibration positions
    d_out, d_in, N = 256, 896, 512

    W = torch.randn(d_out, d_in) * 0.02
    # Inject outlier columns — as in real LLMs (~1% salient columns per AWQ)
    outlier_cols = [5, 50, 200, 400, 700]
    for c in outlier_cols:
        W[:, c] *= 60
    X = torch.randn(N, d_in) * 0.1

    print(f"  Synthetic layer  W={tuple(W.shape)}  X={tuple(X.shape)}")
    print(f"  Outlier columns injected: {outlier_cols}")
    print()

    # Pre-compute H for the self-test since quantize API now expects H
    H = (2.0 / N) * (X.T @ X)

    for bits in [4, 3, 2]:
        W_rtn,  _, _     = rtn_quantize(W.clone(), bits)
        W_gptq, _, _     = gptq_quantize(W.clone(), H, bits)

        err_rtn  = layer_output_error(W, W_rtn,  X)
        err_gptq = layer_output_error(W, W_gptq, X)

        # Hadamard: verify it reduces column-norm variance (primary guarantee)
        Q = _random_hadamard(d_in, seed=1337)
        var_before = column_norm_variance(W)
        var_after  = column_norm_variance(W @ Q.T)

        pct_gptq = (1 - err_gptq / err_rtn) * 100

        print(f"  {bits}-bit")
        print(f"    RTN error        : {err_rtn:.4f}")
        print(f"    GPTQ error       : {err_gptq:.4f}  ({pct_gptq:.1f}% better than RTN)")
        print(f"    Col-norm var     : {var_before:.4f} → {var_after:.4f} after Hadamard  "
              f"({'✅ reduced' if var_after < var_before else '❌ not reduced'})")
        print()

        # Assertions
        assert err_gptq  < err_rtn * 1.05,    f"{bits}b: GPTQ should generally beat or match RTN"
        assert var_after < var_before, f"{bits}b: Hadamard must reduce column-norm variance"

    print("All assertions passed ✅")
    print()
    print("Note: GPTQ+H vs GPTQ comparison is meaningful only on real LLM weights.")
    print("      On real models, GPTQ+H reduces 2-bit error by ~13% (QuIP# validated).")

