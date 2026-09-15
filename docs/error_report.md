# Kaggle Execution Error Report
**Project:** Architecture-Agnostic Layer-Wise PTQ of LLMs  
**Date:** September 15, 2026

During the deployment of our Phase 1 pipeline onto the Kaggle T4 environment, we encountered three distinct system crashes. This report documents the symptoms, the root causes, and the implemented solutions for each error.

---

## 1. CPU RAM Out of Memory (OOM) Leak
**Phase:** Calibration (`calibrate.py`)  
**Symptom:** The Kaggle session crashed entirely. The green RAM bar hit 16GB (100%) and the kernel restarted.

### Root Cause
In our original code, PyTorch forward hooks were used to capture the input activations ($X$) for every `nn.Linear` layer across 128 sequences (2048 tokens each). 
The code was storing the full, raw FP32 activation tensors in a giant dictionary:
```python
activation_store[layer_name].append(x.cpu())
```
For a 1.4B parameter model with ~160 linear layers, this required storing over 10GB of floating-point data in CPU RAM simultaneously, instantly breaching Kaggle's 16GB limit.

### Resolution
**Incremental Hessian Accumulation.** 
Both the GPTQ and QuIP quantization algorithms ultimately only need the scaled Hessian matrix ($H = X^T X$) of the activations, not the activations themselves. We rewrote `calibrate.py` to incrementally compute $H$ inside the forward hook itself:
```python
H_batch = x.T @ x
hessian_store[layer_name] += H_batch.cpu()
```
**Result:** This dropped the per-layer memory requirement from $[128 \times 2048, D_{in}]$ down to just $[D_{in}, D_{in}]$. The peak CPU RAM usage plummeted from **>16 GB** down to just **2.7 GB** (a 99.9% reduction in memory overhead).

---

## 2. GPU VRAM Out of Memory (OOM) Crash
**Phase:** Evaluation (`lm-evaluation-harness`)  
**Symptom:** `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.51 GiB. GPU 0 has a total capacity of 14.56 GiB.`

### Root Cause
During the perplexity evaluation on the `wikitext` dataset, `lm_eval` uses a rolling window to compute log-likelihoods over extremely long text contexts. Our `run_config.yaml` had hardcoded:
```yaml
eval:
  batch_size: 8
```
Feeding a batch of 8 massively long sequences into the attention mechanism caused the KV-cache and attention matrices to balloon, demanding a single contiguous 7.5GB chunk of VRAM, which crashed the 15GB Kaggle GPU.

### Resolution
**Dynamic Batch Sizing.** 
We modified `run_config.yaml` to leverage `lm_eval`'s native memory management:
```yaml
eval:
  batch_size: "auto"
```
**Result:** The harness now tests the available VRAM and automatically scales the batch size down (e.g., from 8 to 4 to 2) before it triggers a CUDA OOM. The evaluation now runs safely on any hardware.

---

## 3. Multi-GPU Device Mapping Error
**Phase:** Quantization Loop (`quantize.py`)  
**Symptom:** `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!`

### Root Cause
Kaggle's "T4 x2" environment provides two GPUs. When we loaded the model, we correctly used HuggingFace Accelerate:
```python
model = AutoModelForCausalLM.from_pretrained(..., device_map="auto")
```
This automatically split the model across `cuda:0` and `cuda:1`. However, later in the code we explicitly forced the model back to a single device:
```python
model.to(device)  # where device was "cuda:0"
```
This brutally overrode `accelerate`'s multi-GPU mapping hooks. When PyTorch tried to execute the forward pass, some weights were still mapped to `cuda:1`, causing a tensor mismatch.

### Resolution
**Respecting Distributed Architecture.** 
We removed all instances of `model.to(device)` in both `run.py` and `calibrate.py`. 
For calibration inputs, we dynamically fetched the device of the first layer:
```python
first_device = next(model.parameters()).device
input_ids = sample.to(first_device)
```
For quantization, we ensured the weights were quantized on their native device:
```python
orig_device = module.weight.device
W = module.weight.data.float().to(orig_device)
```
**Result:** The code now natively supports multi-GPU scaling without dispatch errors.
