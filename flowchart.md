# 🗺️ Project Flowchart — Architecture-Agnostic Layer-Wise PTQ of LLMs
### BTech Final Year Project | Faqre Alam · Guneet Toppo · Ekansh Agrawal | DTU | 2026–27

---

## 1. Big Picture — End-to-End Project Flow

> The complete pipeline from a pretrained model to the final results table.

```mermaid
flowchart TD
    A(["🤖 Pretrained LLM\nFP16 weights  •  0.5B–1.1B params\nQwen2.5-0.5B / Llama-3.2-1B / TinyLlama-1.1B"])

    subgraph CAL["STEP 1 — Calibration"]
        B["Load 128 sequences\n× 2048 tokens from WikiText-2 train"]
        C["Register forward hooks\non every nn.Linear in the model"]
        D["Run one forward pass\n→ capture input activation X\nfor each linear layer"]
        E[("X per layer\nstored on CPU RAM")]
        B --> C --> D --> E
    end

    subgraph QUANT["STEP 2 — Quantization  (choose method + bits)"]
        F["RTN\n2 / 3 / 4 bit"]
        G["GPTQ\n2 / 3 / 4 bit"]
        H["GPTQ + Hadamard\n2 / 3 / 4 bit"]
        I["Mixed-Precision\nHessian-Diagonal {2,4}-bit\n⭐ Novel Contribution"]
    end

    subgraph OUT["STEP 3 — Quantized Model"]
        J["Layer weights replaced\nINT2 / INT3 / INT4\nModel size: 125 MB – 550 MB"]
    end

    subgraph EVAL["STEP 4 — Evaluation  via lm-evaluation-harness"]
        K["WikiText-2 Perplexity\n(language quality)"]
        L["ARC-Easy  0-shot\n(factual knowledge)"]
        M["HellaSwag  0-shot\n(language coherence)"]
        N["WinoGrande  0-shot\n(commonsense reasoning)"]
    end

    R(["📋 Results Table\n3 methods × 3 bits × 3 families = 27 runs\n+ Mixed-Precision sweep"])

    A --> CAL
    CAL --> QUANT
    E -. activations used by .-> QUANT
    F & G & H & I --> J
    J --> EVAL
    K & L & M & N --> R

    style A fill:#1a1a2e,color:#e0e0ff,stroke:#7c83fd
    style R fill:#1a1a2e,color:#e0e0ff,stroke:#7c83fd
    style I fill:#2d1b69,color:#e9d5ff,stroke:#a78bfa
    style J fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

---

## 2. Calibration Data Pipeline

> Captures the per-layer activations X that ALL quantization methods depend on.

```mermaid
flowchart TD
    A["WikiText-2  — Train Split\n~2 million tokens  (plain text)"]
    B["Model Tokenizer\n(loaded separately per model family)"]
    C["Long Token Tensor\nshape: [1 × total_tokens]"]
    D["Random Window Sampler\npick 128 non-overlapping windows\neach of length 2048 tokens\nusing fixed seed=42"]
    E["Calibration Tensor\nshape: [128 × 2048]\ndtype: int64  (token IDs)"]
    F["Register Forward Hook\non every nn.Linear\nhook records:  input[0]  → X"]
    G["Single Forward Pass\nthrough the full model\ntorch.no_grad()  —  no gradients"]
    H["Hook fires for each layer\ncollects X_layer on CPU"]
    I[("Activation Store\ndict:  layer_name → X\nX shape: [128×2048 × d_in]\nkept on CPU to save GPU RAM")]
    J["Remove all hooks\n(clean up before quantization)"]

    A --> B --> C --> D --> E
    E --> F
    F --> G --> H --> I
    I --> J

    style A fill:#1c3a5e,color:#bae6fd,stroke:#38bdf8
    style I fill:#1c3a5e,color:#bae6fd,stroke:#38bdf8
    style J fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

> **Why WikiText-2?** Domain-neutral, universally used, fixes the calibration distribution — so all 27 runs are directly comparable. Different calibration data shifts perplexity by 0.1–0.5 points.

---

## 3. Method 1 — RTN  (Round-To-Nearest Baseline)

> The simplest possible quantization — no calibration data used at all.

```mermaid
flowchart TD
    A["Input:\nWeight Matrix W\nshape [d_out × d_in]  dtype FP16"]

    subgraph PER["Per-channel  (repeat for each output channel row)"]
        B["Find range:\nw_min = min(W[i,:])  •  w_max = max(W[i,:])"]
        C["Compute scale:\ns = (w_max − w_min) / (2^b − 1)"]
        D["Compute zero-point:\nz = round(−w_min / s)\nclipped to [0, 2^b−1]"]
        E["Quantize:\nW_q = clamp( round(W / s) + z,  0,  2^b−1 )\ndtype: UINT2 / UINT3 / UINT4"]
        B --> C --> D --> E
    end

    F["Store compressed weights:\n• W_q  as packed INT\n• scale s  as FP16  (one per channel)\n• zero-point z  as INT8"]
    G["At inference — dequantize on-the-fly:\nW̃ = s × (W_q − z)  back to FP16\nthen compute  Y = W̃ · X  normally"]

    A --> PER --> F --> G

    style A fill:#3b1f00,color:#fde68a,stroke:#f59e0b
    style F fill:#3b1f00,color:#fde68a,stroke:#f59e0b
    style G fill:#1a1a2e,color:#e0e0ff,stroke:#7c83fd
```

**Properties:**
| | |
|:---|:---|
| ✅ No calibration data | Zero-shot compression |
| ✅ O(d_out × d_in) | Fastest method |
| ❌ No error correction | Brutal at 2-bit |
| ❌ Quant error = s²/12 | Grows fast as bits decrease |

---

## 4. Method 2 — GPTQ  (Second-Order Error Compensation)

> Uses calibration activations to build a Hessian, then compensates for each column's quantization error by adjusting remaining unquantized columns.

```mermaid
flowchart TD
    A["Input:\nWeight W [d_out × d_in]  FP16\nCalibration activations X [N × d_in]"]

    subgraph HESS["Hessian Construction  (one-time per layer)"]
        B["H = 2 · Xᵀ · X\nshape [d_in × d_in]\nThis is the Fisher information proxy"]
        C["Add dampening for numerical stability:\nH ← H + ε·I\nwhere ε = 0.01 × mean(diag H)"]
        D["Cholesky decomposition:\nH = L · Lᵀ\nInvert H column-by-column via back-substitution\n→ avoids full O(n³) inversion,  numerically stable"]
        B --> C --> D
    end

    subgraph LOOP["Column-wise Quantization Loop  j = 1 … d_in"]
        E["Quantize column j:\nFind scale s_j from W[:,j] range\nW_q[:,j] = round(W[:,j] / s_j) × s_j"]
        F["Compute quantization error:\nerr[:,j] = W[:,j] − W_q[:,j]"]
        G["Compensate ALL remaining columns j+1 … d_in:\nW[:, j+1:] −= err[:,j] · H⁻¹[j, j+1:] / H⁻¹[j,j]\n(exact minimizer of residual reconstruction loss)"]
        H{"j = d_in\n?"}
        E --> F --> G --> H
        H -->|"No — next column"| E
    end

    I["Output:\nQuantized W with minimal\nreconstruction error\nAll compensation done in FP16"]

    A --> HESS --> LOOP
    H -->|"Yes — done"| I

    style B fill:#1a3a1a,color:#bbf7d0,stroke:#4ade80
    style D fill:#1a3a1a,color:#bbf7d0,stroke:#4ade80
    style G fill:#1a3a1a,color:#bbf7d0,stroke:#4ade80
    style I fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

> **Key insight:** Quantizing column j introduces error `err`. The update `W[:, j+1:] -= err · H⁻¹[j, j+1:] / H⁻¹[j,j]` is the *exact* closed-form solution that minimises the residual squared reconstruction error for all future columns simultaneously.

---

## 5. Method 3 — GPTQ + Hadamard Incoherence

> Before applying GPTQ, rotate the weight matrix so that no column is an outlier. This spreads quantization pain uniformly — critical for 2-bit.

```mermaid
flowchart TD
    A["Input:\nWeight W [d_out × d_in]\nCalibration activations X [N × d_in]"]

    subgraph WHY["Why rotation is needed"]
        B["Inspect column norms:\nsome columns have ‖W[:,j]‖ >> average\nthese outlier columns absorb disproportionate\nquantization error at 2-bit"]
    end

    subgraph ROT["Randomised Hadamard Rotation  (offline, one-time)"]
        C["Sample random sign vector s ∈ {±1}^d_in\nForm randomised Hadamard matrix:\nQ = (1/√d_in) · Diag(s) · H_Walsh"]
        D["Rotate weights:\nW' = W · Qᵀ\nColumn norms now ≈ uniform\nNo single column dominates"]
        E["Rotate calibration activations:\nX' = X · Q\n(mathematically equivalent:  W·X = W'·X')"]
        C --> D --> E
    end

    subgraph APPLY["Apply standard GPTQ on rotated tensors"]
        F["Run GPTQ on (W', X')\nas described in Method 2\nQuantized result: W'_q"]
    end

    subgraph STORE["Storage & Inference"]
        G["Store:\n• Quantized W'_q  (INT2/3/4)\n• Rotation seed only  (not the full Q matrix!)"]
        H["At inference time:\nReconstruct Q from seed\nApply X' = X · Q  before matmul\nCost: O(d_in log d_in) — negligible"]
    end

    A --> WHY --> ROT --> APPLY --> STORE

    style C fill:#2d1b69,color:#e9d5ff,stroke:#a78bfa
    style D fill:#2d1b69,color:#e9d5ff,stroke:#a78bfa
    style G fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

**Effect of Hadamard rotation:**
```
BEFORE  col7: [0.003, 0.002, ..., 148.7, 0.001]   ← outlier!
AFTER   col7: [0.81,  0.79,  ...,  0.82, 0.80 ]   ← uniform ✅

Guarantee:  E[max_j ‖W'[:,j]‖²]  ≈  ‖W‖²_F / d_in   (concentration bound)
```

---

## 6. ⭐ Novel Contribution — Hessian-Diagonal Mixed-Precision Allocation

> Assign different bit-widths to different weight columns based on their *quantization sensitivity* — all within a fixed average bit budget.

```mermaid
flowchart TD
    A["Input:\nW [d_out × d_in]  +  X [N × d_in]\nTarget average bits: b̄ = 3\nBit choices: {2-bit, 4-bit}"]

    subgraph SENS["Sensitivity Scoring  (O(N·d_in) — no matrix inversion!)"]
        B["Hessian diagonal:\nH_jj = 2 · Σ_n X[n,j]²\nMeasures: how much output changes when column j shifts"]
        C["Column weight variance:\nVar_j = Var(W[:,j])\nMeasures: how hard column j is to quantize"]
        D["Sensitivity score:\nŝ_j = H_jj × Var_j\nfor each j = 1 … d_in\n\nDerived from 2nd-order Taylor expansion:\nE[ΔLoss_j] ∝ H_jj × Var_j / 4^b"]
        B --> D
        C --> D
    end

    subgraph ALLOC["Bit Allocation (Reverse Water-Filling)"]
        E["Sort all d_in columns by ŝ_j  (descending)"]
        F["At b̄=3 with {2,4}-bit only:\nt = d_in / 2  columns get 4-bit\n(d_in − t) columns get 2-bit\n→ average = (4t + 2(d_in−t)) / d_in = 3 ✅"]
        G{"Sweep:\nvary t from\n10% to 90%\nof d_in"}
        E --> F --> G
    end

    subgraph QSTEP["Quantize Each Group"]
        H["Apply GPTQ on top-t columns\nusing 4-bit precision"]
        I["Apply GPTQ on bottom-(d_in−t) columns\nusing 2-bit precision"]
        H & I --> J
    end

    J["Output:\nMixed-precision quantized W\nSame memory as uniform 3-bit\nLower reconstruction error than uniform 3-bit"]

    A --> SENS --> ALLOC --> QSTEP

    style D fill:#4a1942,color:#f5d0fe,stroke:#d946ef
    style F fill:#4a1942,color:#f5d0fe,stroke:#d946ef
    style J fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

**The budget comparison at equal average bits:**
```
Config                 Bits per column              Avg bits   Memory
─────────────────────────────────────────────────────────────────────
All 2-bit              [2, 2, 2, 2, 2, 2, 2, 2]     2.0 bits   ~125 MB
Uniform 3-bit  ←base   [3, 3, 3, 3, 3, 3, 3, 3]     3.0 bits   ~190 MB
Your method    ←novel   [4, 4, 4, 4, 2, 2, 2, 2]     3.0 bits   ~190 MB  ✅ better PPL
All 4-bit              [4, 4, 4, 4, 4, 4, 4, 4]     4.0 bits   ~250 MB
                        ↑ sensitive     ↑ insensitive
```

---

## 7. The 27-Run Ablation Grid

> The controlled cross-architecture study that is the core empirical contribution.

```mermaid
flowchart LR
    subgraph MODELS["3 Model Families"]
        M1["Qwen2.5-0.5B\nAlibaba · GQA 7:1 · 24 layers"]
        M2["Llama-3.2-1B\nMeta · GQA 4:1 · 16 layers"]
        M3["TinyLlama-1.1B\nOpen · Full MHA · 22 layers"]
    end

    subgraph METHODS["3 Quantization Methods"]
        T1["RTN\n(no calibration)"]
        T2["GPTQ\n(Hessian compensation)"]
        T3["GPTQ + Hadamard\n(rotation + compensation)"]
    end

    subgraph BITS["3 Bit-widths"]
        B1["2-bit  (8 levels)"]
        B2["3-bit  (8→16 eff. with group)"]
        B3["4-bit  (16 levels)"]
    end

    CROSS{"3 × 3 × 3\n= 27 runs\n+ 3 FP16 baselines"}

    subgraph METRICS["4 Metrics per run"]
        R1["WikiText-2 PPL ↓"]
        R2["ARC-Easy Acc ↑"]
        R3["HellaSwag Acc ↑"]
        R4["WinoGrande Acc ↑"]
    end

    MODELS --> CROSS
    METHODS --> CROSS
    BITS --> CROSS
    CROSS --> METRICS

    style CROSS fill:#2d1b69,color:#e9d5ff,stroke:#a78bfa
```

**Expected quality ordering at each bit-width:**

| Bit-width | RTN | GPTQ | GPTQ+H |
|:---:|:---:|:---:|:---:|
| 4-bit | ⚠️ slight drop | ✅ near-lossless | ✅ near-lossless |
| 3-bit | ❌ noticeable | ✅ good | ✅ good |
| 2-bit | 💀 catastrophic | ⚠️ structured loss | ✅ best (~13% better than GPTQ) |

---

## 8. Evaluation Pipeline

> How each of the 27 quantized models is measured after compression.

```mermaid
flowchart TD
    A["Quantized Model saved to disk\nmodel.save_pretrained(path)\nweights: INT2/3/4 + scale metadata"]

    B["Load model into HuggingFace pipeline\nAutoModelForCausalLM.from_pretrained(path)\ndequantize on-the-fly at runtime"]

    C["EleutherAI lm-evaluation-harness  v0.4.x\n(pinned version for reproducibility)\nlm_eval --model hf --model_args pretrained=path"]

    subgraph TASKS["Parallel Task Evaluation"]
        direction LR
        D1["wikitext\n─────────\nMetric: Perplexity\nexp(−mean log P)\nTest split, stride=512\nLower = better ↓"]
        D2["arc_easy\n─────────\nMetric: 0-shot Accuracy\n2 answer choices\nNormalized log-likelihood\nHigher = better ↑"]
        D3["hellaswag\n─────────\nMetric: 0-shot Accuracy\n4 sentence completions\nNormalized log-likelihood\nHigher = better ↑"]
        D4["winogrande\n─────────\nMetric: 0-shot Accuracy\n2-way pronoun resolution\nHigher = better ↑"]
    end

    E["JSON output per config:\n{ model, method, bits, ppl, arc, hellaswag, winogrande }"]

    F["Aggregate all 27 JSONs\n→ Final Results Table CSV"]

    A --> B --> C --> TASKS
    D1 & D2 & D3 & D4 --> E --> F

    style C fill:#1e3a5f,color:#bfdbfe,stroke:#60a5fa
    style F fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

---

## 9. Architecture Comparison — Why These 3 Families?

> Each family differs in how attention is structured, making them non-trivial to compare.

```mermaid
flowchart TD
    subgraph TINYLLAMA["TinyLlama-1.1B  ·  Llama-2 style"]
        T1["Params: 1.1B\nHidden dim: 2048\nLayers: 22\nFFN hidden: 5632\nAttention: 32 Q-heads / 32 KV-heads  (Full MHA)\nNo GQA  →  every head has its own K, V\nEmbedding: tied\nRoPE + RMSNorm + SwiGLU"]
    end

    subgraph LLAMA["Llama-3.2-1B  ·  Llama-3 style"]
        L1["Params: 1.24B\nHidden dim: 2048\nLayers: 16\nFFN hidden: 8192\nAttention: 32 Q-heads / 8 KV-heads  (GQA 4:1)\n4 query heads share 1 KV head\nEmbedding: not tied\nRoPE + RMSNorm + SwiGLU"]
    end

    subgraph QWEN["Qwen2.5-0.5B  ·  Alibaba"]
        Q1["Params: 0.5B\nHidden dim: 896\nLayers: 24\nFFN hidden: 4864\nAttention: 14 Q-heads / 2 KV-heads  (GQA 7:1)\n7 query heads share 1 KV head  (very aggressive)\nEmbedding: tied\nRoPE + RMSNorm + SwiGLU"]
    end

    SHARED["All 3 share:\nDecoder-only transformer  ·  RoPE  ·  RMSNorm  ·  SwiGLU  ·  Apache 2.0\nAll loaded via  AutoModelForCausalLM.from_pretrained()\nAll quantized by the EXACT SAME hook-based framework\n→ proves architecture-agnostic claim"]

    TINYLLAMA --> SHARED
    LLAMA --> SHARED
    QWEN --> SHARED

    style SHARED fill:#1a1a2e,color:#e0e0ff,stroke:#7c83fd
    style TINYLLAMA fill:#1c2a1c,color:#bbf7d0,stroke:#4ade80
    style LLAMA fill:#1c2a1c,color:#bbf7d0,stroke:#4ade80
    style QWEN fill:#1c2a1c,color:#bbf7d0,stroke:#4ade80
```

**Research question encoded here:**
> GQA ratio (7:1 vs 4:1 vs none) changes the parameter count and activation profile of `k_proj` and `v_proj`. Does the Hessian-diagonal sensitivity score correctly identify these differences — without any architecture-specific code?

---

## 10. Memory & Compute Budget on Kaggle T4

```mermaid
flowchart LR
    subgraph GPU["Kaggle T4  (Free)"]
        direction TB
        G1["VRAM: 16 GB"]
        G2["Compute: ~8 TFLOPS FP32"]
        G3["Quota: 30 GPU-hrs / week"]
    end

    subgraph MEM["Memory breakdown per run\n(TinyLlama worst case)"]
        direction TB
        M1["Model weights FP16: 2.2 GB on GPU"]
        M2["GPTQ Hessian per layer: ~64 MB on GPU\n(loaded one layer at a time)"]
        M3["Calibration activations: ~2 GB on CPU\n(offloaded — never on GPU all at once)"]
        M4["Peak GPU usage: ~5 GB out of 16 GB  ✅"]
    end

    subgraph TIME["Timing per run"]
        direction TB
        T1["Calibration forward pass: 2–4 min"]
        T2["GPTQ layer-wise loop: 15–25 min"]
        T3["lm-eval 4 tasks: 15–20 min"]
        T4e["Total per run: ~35–50 min"]
        T5["27 runs total: ~18 GPU-hours  ✅\nfits in Kaggle weekly quota"]
    end

    GPU --> MEM
    GPU --> TIME

    style GPU fill:#1c3a5e,color:#bae6fd,stroke:#38bdf8
```

| Model | Calibration | GPTQ | Eval | **Total/run** |
|:---|:---:|:---:|:---:|:---:|
| Qwen2.5-0.5B | 2 min | 15 min | 15 min | **~32 min** |
| Llama-3.2-1B | 3 min | 15 min | 20 min | **~38 min** |
| TinyLlama-1.1B | 4 min | 25 min | 20 min | **~49 min** |
| **27 runs total** | | | | **~18 GPU-hours ✅** |

---

## 11. Complete Low-Level Implementation Data Flow

> Exact code-level execution order for Phase 1.

```mermaid
flowchart TD
    A(["HuggingFace Hub\nmodel checkpoint"])

    subgraph LOAD["Model Loading"]
        B["AutoModelForCausalLM.from_pretrained\ndtype=torch.float16\ndevice_map=auto"]
    end

    subgraph CALIB["Calibration Phase"]
        C["For each nn.Linear in model:\nmodule.register_forward_hook(capture_fn)\ncapture_fn saves input[0] to dict on CPU"]
        D["Run all 128 calibration sequences\nthrough model.forward()  no gradients"]
        E["For each module:\nstack collected inputs → X_layer\nshape [N_total × d_in]"]
        F["Remove all hooks\nclean up memory"]
        C --> D --> E --> F
    end

    subgraph QLOOP["Layer-wise Quantization Loop"]
        G["for name, module in model.named_modules():"]
        H{"module is\nnn.Linear ?"}
        SKIP["skip — do nothing\n(embeddings, norms, etc.)"]
        I["Load X = activation_store[name] to GPU"]
        J["Compute H = 2 · Xᵀ · X  on GPU"]
        K["Apply chosen method:\nRTN(W, bits)\nor GPTQ(W, H, bits)\nor GPTQ_H(W, H, bits)"]
        L["module.weight.data ← W_quantized\nin-place replacement"]
        M["del X, H  — free GPU memory"]
        N["continue to next module"]
        G --> H
        H -->|"No"| SKIP --> N
        H -->|"Yes"| I --> J --> K --> L --> M --> N
        N -->|"more modules"| G
    end

    subgraph SAVE["Save & Evaluate"]
        O["model.save_pretrained(output_path)\ntokenizer.save_pretrained(output_path)"]
        P["lm_eval --model hf\n--model_args pretrained=output_path\n--tasks wikitext,arc_easy,hellaswag,winogrande\n--device cuda:0  --batch_size 8"]
        Q["Parse JSON results\nappend row to ablation_table.csv"]
        O --> P --> Q
    end

    A --> LOAD --> CALIB --> QLOOP
    N -->|"all done"| SAVE

    style J fill:#1a3a1a,color:#bbf7d0,stroke:#4ade80
    style K fill:#1a3a1a,color:#bbf7d0,stroke:#4ade80
    style Q fill:#0d3b2e,color:#d4f5e9,stroke:#34d399
```

---

## 12. Your Novel Method vs. Prior Work

```mermaid
flowchart LR
    subgraph HAWQ["HAWQ-V2 / V3  (Prior Art)"]
        H1["Sensitivity: Hessian TRACE  (layer-level)\nGranularity: per LAYER\nSolver: ILP  (expensive)\nDomain: CNNs only\nCross-arch: ❌ No"]
    end

    subgraph AWQ["AWQ  (Prior Art)"]
        A1["Sensitivity: activation MAGNITUDE  (1st-order)\nGranularity: per CHANNEL\nSolver: grid search\nDomain: Llama-centric\nCross-arch: ❌ No"]
    end

    subgraph QBERT["Q-BERT  (Prior Art)"]
        QB1["Sensitivity: Hessian  (layer-level)\nGranularity: per LAYER\nDomain: BERT encoder only\nDecoder: ❌ No"]
    end

    subgraph YOURS["⭐ YOUR METHOD  (Novel)"]
        Y1["Sensitivity: Hessian DIAGONAL × weight Var  (2nd-order)\nGranularity: per CHANNEL  (finer than HAWQ)\nSolver: NONE  — closed-form O(mn)\nDomain: Decoder LLMs\nCross-arch: ✅ 3 families tested\nBit format: software-only INT {2,4}"]
    end

    HAWQ -->|"+ finer granularity\n+ no ILP solver\n+ LLM domain"| YOURS
    AWQ -->|"+ 2nd-order math\n+ principled vs heuristic"| YOURS
    QBERT -->|"+ decoder-only\n+ cross-arch study"| YOURS

    style YOURS fill:#2d1b69,color:#e9d5ff,stroke:#a78bfa
    style HAWQ fill:#2a1a1a,color:#fecaca,stroke:#f87171
    style AWQ fill:#2a1a1a,color:#fecaca,stroke:#f87171
    style QBERT fill:#2a1a1a,color:#fecaca,stroke:#f87171
```

---

## 13. Expected Results  (Qualitative)

### Perplexity at 2-bit  (lower = better)

```
FP16 baseline       ████  ~7 PPL

RTN 2-bit           ████████████████████████████████████  100+ PPL  💀 catastrophic

GPTQ 2-bit          ██████████████████  40–60 PPL          ⚠️ large but structured

GPTQ+Hadamard 2-bit █████████████  25–35 PPL              ✅ ~13% better than GPTQ

Mixed {2,4} avg=3b  █████████  15–20 PPL                  ⭐ your method — best!

Uniform 3-bit GPTQ  ████████  12–15 PPL                   ← direct comparison baseline
```

### Cross-architecture transfer — what the 27-run table will reveal

```mermaid
flowchart LR
    A["Run GPTQ+H on all 3 families\nat 2-bit"]
    B{"Is PPL improvement\nsimilar across families?"}
    C["Method TRANSFERS\nHadamard benefit is\narchitecture-independent\n→ it's a property of the math"]
    D["Architecture-SPECIFIC\nBenefit depends on\nweight distribution\nor GQA structure\n→ empirical finding"]

    A --> B
    B -->|"Yes — similar"| C
    B -->|"No — varies"| D
```

---

## 14. GitHub Repository Structure

```
final-year-btp/
│
├── flowchart.md                    ← this file
├── README.md                       ← project overview + main results table
│
├── phase0/                         ✅ COMPLETE
│   ├── demo_quant.py               ← NumPy RTN + GPTQ + Hadamard (~200 lines)
│   ├── math_proofs.pdf             ← grid error lemma, water-filling derivation
│   └── phase0_report.pdf           ← 6-page compiled LaTeX report
│
├── phase1/                         🔄 IN PROGRESS  (Oct–Dec 2026)
│   ├── framework/
│   │   ├── calibrate.py            ← hook-based activation capture
│   │   ├── quantize.py             ← RTN, GPTQ, GPTQ+H  (PyTorch, arch-agnostic)
│   │   └── evaluate.py             ← lm-eval wrapper + JSON → CSV
│   ├── configs/
│   │   └── run_config.yaml         ← model IDs, bit-widths, methods, random seed
│   └── results/
│       └── ablation_table.csv      ← 27-run results + FP16 baselines
│
├── phase2/                         📅 Jan–Feb 2027
│   ├── mixed_precision.py          ← Hessian-diagonal allocation algorithm
│   ├── sweep.py                    ← vary t (10%→90%), trace PPL vs avg-bits curve
│   └── results/
│       └── mixedprec_results.csv
│
├── phase3/                         📅 Mar–May 2027
│   ├── paper/
│   │   ├── main.tex                ← ACL-style LaTeX paper
│   │   └── figures/                ← all plots + result tables
│   └── analysis/
│       └── plot_results.py         ← generate all paper figures from CSVs
│
└── data/
    └── calibration/
        └── wikitext2_calib_seed42.pt   ← saved calibration tensors
                                           load once, reuse across all 27 runs
```

---

*Compiled: September 2026 · Faqre Alam · Guneet Toppo · Ekansh Agrawal · BTech Computer Engineering · Delhi Technological University*
