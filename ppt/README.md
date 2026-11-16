# DTU CSE BTP GP159 — HTML Presentation Slide Deck

Interactive, web-native presentation for **Major Project (BTP GP159)**:
> **"Task-Conditioned Evaluation of Quantized Small Language Models"**  
> **Delhi Technological University (DTU)** | CSE Department | Session 2026-27  
> **Guide:** Ms. Gull Kaur, Assistant Professor  
> **Team:** Ekansh Agarwal (23/CS/149), Faqre Alam (23/CS/150), Guneet Toppo (23/CS/160)

---

## 🚀 How to Run the Presentation

### Option 1: Double-click or Terminal (Mac/Linux/Windows)
Simply open `ppt/index.html` in **Google Chrome**, **Safari**, or **Edge**:

```bash
# On macOS:
open ppt/index.html

# On Linux:
xdg-open ppt/index.html

# On Windows:
start ppt/index.html
```

### Option 2: Local Web Server (Recommended for smooth rendering)
```bash
python3 -m http.server 8000
# Open http://localhost:8000/ppt/ in your browser
```

---

## ⌨️ Presentation Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`Right Arrow`** / **`Space`** / **`Enter`** / **`PageDown`** | Next slide |
| **`Left Arrow`** / **`Backspace`** / **`PageUp`** | Previous slide |
| **`Home`** | Jump to Title slide (Slide 1) |
| **`End`** | Jump to Conclusion & Defense slide (Slide 16) |
| **Touch / Mobile / iPad** | Swipe Left (Next) / Swipe Right (Prev) |
| **Navigation Bar** | Click `PREV`, `NEXT`, or `FULLSCREEN` buttons |

---

## 📋 Complete Slide Outline (16 Slides)

1. **Title Slide**: Project details, DTU GP159, Team members & roles, Guide Ms. Gull Kaur.
2. **Executive Summary & The "Perplexity Illusion"**: Why 4-bit models seem intact on text perplexity (+0.95) but collapse downstream.
3. **Problem Statement & Literature Gap**: Prior works (Dettmers, Frantar) vs SLM reality gap; Stanford CRFM HELM grounding.
4. **Research Questions**: Formal breakdown of RQ1 (Cognitive sensitivity), RQ2 (Rank inversion), and RQ3 (Noise-aware bit-budget).
5. **End-to-End System Architecture Flowchart**: Hugging Face $\rightarrow$ PTQ Quantizer $\rightarrow$ Capped Batching Runner $\rightarrow$ LM-Eval $\rightarrow$ Atomic Store.
6. **Mathematical Overview of Evaluated PTQ Methods**: FP16, BitsAndBytes (LLM.int8() & NF4), GPTQ (inverse Hessian), and AWQ (activation salience).
7. **5-Tier Cognitive Benchmark Taxonomy**: 7 tasks mapped to 5 cognitive tiers (WikiText $\rightarrow$ ARC/HellaSwag $\rightarrow$ WinoGrande $\rightarrow$ MMLU $\rightarrow$ GSM8K/IFEval).
8. **Experimental Matrix & Hardware Grid**: 3 architectures (Qwen2.5-0.5B, Llama-3.2-1B, TinyLlama-1.1B), 15 configurations, 105 task evaluations on Kaggle T4 cloud GPUs.
9. **Empirical Results (Tiers 1–4)**: Word perplexity and multiple-choice accuracy drops across 8-bit and 4-bit variants.
10. **Empirical Results (Tier 5 Collapse)**: Generative reasoning cliff (GSM8K 18+ pp drop) and instruction following drift (IFEval).
11. **Quantization Method Showdown**: AWQ vs GPTQ vs BitsAndBytes NF4 — why AWQ preserves critical attention routing.
12. **Stanford CRFM HELM Metrics**: Beyond accuracy: Expected Calibration Error (ECE) and Brier score revealing overconfident hallucination at 4-bit.
13. **Statistical Rigor**: 10,000-iteration paired bootstrap confidence intervals (BCa) proving statistical significance ($p < 0.0001$).
14. **Rank Inversion & Spearman $\rho$ Analysis**: Proving benchmark rank order stability degrades on reasoning tasks ($\rho = 0.618$).
15. **Engineering Deliverable**: The noise-aware bit-budget decision framework ($\Delta \le 3.0\text{ pp}$ safety bounds).
16. **Conclusion, Key Contributions & Defense Ready**: 4 core contributions, open-source repository, ready for viva voce defense.
