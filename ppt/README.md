# DTU CSE BTP GP159 — Presentation Slide Deck (Concise & Intuitive)

Interactive, web-native presentation for **Major Project (BTP GP159)**:
> **"Task-Conditioned Evaluation of Quantized Small Language Models"**  
> **Delhi Technological University (DTU)** | CSE Department  
> **Guide:** Ms. Gull Kaur, Assistant Professor  
> **Team:** Ekansh Agarwal (23/CS/149), Faqre Alam (23/CS/150), Guneet Toppo (23/CS/160)

---

## 🚀 How to Open the Presentation

Simply open [`ppt/index.html`](file:///Users/faqrealam149/Desktop/final-year-btp/ppt/index.html) in your browser:

```bash
# On macOS:
open ppt/index.html

# Or via Python local server:
python3 -m http.server 8000
# Open http://localhost:8000/ppt/
```

---

## ⌨️ Presentation Controls

- **`Right Arrow` / `Space` / `Enter`**: Next Slide
- **`Left Arrow` / `Backspace`**: Previous Slide
- **`F` / Fullscreen Button**: Toggle Fullscreen Mode
- **Touch / Trackpad**: Swipe Left (Next) / Swipe Right (Prev)

---

## 📋 The 12-Slide Intuitive Storyline

1. **Title Slide**: Project details, DTU CSE GP159, Guide Ms. Gull Kaur, Team members (Ekansh, Faqre, Guneet).
2. **What is Quantization? (In Plain Words)**: Video compression analogy, 16-bit float $\rightarrow$ 8-bit $\rightarrow$ 4-bit bins.
3. **Why is it Important?**: 75% RAM reduction, 2x–3x speedup, privacy, battery life, zero cloud costs.
4. **The Problem & Why We Are Doing This**: The "Perplexity Illusion" trap — why small models (<2B) break on downstream reasoning despite fluent text.
5. **Our Thinking**: "Not all tasks suffer equally" — single-step recall vs multi-step chain-of-thought compounding errors.
6. **Our Approach**: The 5-Tier Evaluation Ladder (Fluency $\rightarrow$ Facts $\rightarrow$ Logic $\rightarrow$ Broad Knowledge $\rightarrow$ Multi-step Math & Rules).
7. **Parameters & Formulas**: The quantization mapping formula ($q = \text{round}(W/S) + Z$), error explanation, and the 4 methods evaluated (INT8, NF4, GPTQ, AWQ).
8. **The Process We Followed**: Step-by-step pipeline from Hugging Face to Capped Batching, Kaggle T4 cloud grid, and live result saving.
9. **Evaluation Results**: Clear comparison table across Tiers 1–4 vs Tier 5 — proving 8-bit safety and the 4-bit reasoning collapse.
10. **The Confidence Crisis & Statistical Validation**: Expected Calibration Error (ECE) spike ("confidently wrong" model) and 10,000-iteration Bootstrap CIs.
11. **The Engineering Rulebook**: Actionable decision table for developers deploying SLMs on consumer hardware.
12. **Conclusion & Key Takeaways**: Summary of contributions and defense Q&A readiness.
