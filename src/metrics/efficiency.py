"""
efficiency.py
HELM-aligned operational efficiency profiling:
- Model checkpoint disk footprint (MB)
- Peak GPU VRAM allocated during forward & generation passes (MB)
- Generation latency (ms) and throughput (tokens/second)
"""

import os
import time
from typing import Dict, Any, Optional
import torch


def measure_disk_size(path_or_dir: str) -> float:
    """
    Computes total file size in megabytes (MB) for a file or directory on disk.
    """
    if not os.path.exists(path_or_dir):
        return 0.0

    if os.path.isfile(path_or_dir):
        return os.path.getsize(path_or_dir) / (1024 * 1024)

    total_bytes = 0
    for root, _, files in os.walk(path_or_dir):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_bytes += os.path.getsize(fp)

    return total_bytes / (1024 * 1024)


def measure_efficiency(
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
    prompt_tokens: int = 128,
    gen_tokens: int = 64,
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> Dict[str, float]:
    """
    Profiles inference memory and latency on the target hardware.

    Args:
        model: HuggingFace PreTrainedModel or quantized wrapped model
        tokenizer: Corresponding tokenizer
        device: 'cuda', 'mps', or 'cpu'
        prompt_tokens: Fixed length of synthetic prompt to feed
        gen_tokens: Number of tokens to generate
        warmup_runs: Burn-in iterations before timing
        timed_runs: Number of iterations to average latency over

    Returns:
        Dictionary with peak_vram_mb, latency_ms, throughput_tokens_per_sec.
    """
    is_cuda = torch.cuda.is_available() and "cuda" in device

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Generate synthetic prompt of fixed token length
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    # Use deterministic token IDs well within vocabulary
    dummy_input_ids = torch.randint(100, min(1000, vocab_size - 1), (1, prompt_tokens))
    if hasattr(model, "device"):
        target_device = model.device
    else:
        target_device = torch.device(device if is_cuda else "cpu")

    dummy_input_ids = dummy_input_ids.to(target_device)
    attention_mask = torch.ones_like(dummy_input_ids).to(target_device)

    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                input_ids=dummy_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_tokens,
                min_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
            )

    if is_cuda:
        torch.cuda.synchronize(device)

    # Timed runs
    elapsed_times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids=dummy_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_tokens,
                min_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
            )
        if is_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        elapsed_times.append(t1 - t0)

    avg_time_sec = float(sum(elapsed_times) / len(elapsed_times))
    latency_ms = avg_time_sec * 1000.0
    throughput = float(gen_tokens / avg_time_sec) if avg_time_sec > 0 else 0.0

    peak_vram_mb = 0.0
    if is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_mb = float(peak_bytes / (1024 * 1024))

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "latency_ms": latency_ms,
        "throughput_tokens_per_sec": throughput,
        "peak_vram_mb": peak_vram_mb,
    }

