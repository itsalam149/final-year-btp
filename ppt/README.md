# DTU CSE BTP GP159 — Presentation Slide Deck

Interactive, web-native presentation for **Major Project (BTP GP159)**:
> **"Privacy-First Offline AI Mobile Agent Using LLM Quantization"**  
> *(JARVIS-Like On-Device Intelligent Mobile Assistant)*  
> **Delhi Technological University (DTU)** | Department of Computer Science & Engineering  
> **Supervisor:** Ms. Gull Kaur, Assistant Professor  
> **Team:** Faqre Alam (23/CS/150), Ekansh Agarwal (23/CS/149), Guneet Toppo (23/CS/160)

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

## 📋 The 13-Slide Intuitive Storyline

1. **Title Slide**: Project details, DTU CSE GP159, Guide Ms. Gull Kaur, Team members (Faqre, Ekansh, Guneet).
2. **Vision & Philosophy**: "Your personal AI assistant should understand, reason, and perform useful actions directly on your smartphone without leaking a single byte of private data."
3. **The Problem with Cloud Assistants**: Privacy leakage, zero offline availability, latency, recurring subscription costs, and the generation vs. execution gap.
4. **Hardware Boundaries**: Mobile RAM ceiling (&le; 3.5 GB for apps), KV-cache explosion, and thermal/battery boundaries on ARM silicon.
5. **Core Research Discovery**: The "4-Bit Syntax Collapse" — why 4-bit quantized SLMs break on JSON schemas and tool calling despite fluent English.
6. **System Architecture**: The complete 9-module on-device pipeline from offline speech (Whisper) to ReAct planning, llama.cpp GGUF, tool sandbox, and native TTS.
7. **Quantization & GGUF Pipeline**: Precision hierarchy comparison across FP16, Q8_0, Q4_K_M, and Q4_0 on ARM NEON.
8. **Grammar-Constrained Decoding (BNF Grammars)**: The finite-state logit masking engine that mathematically eliminates 4-bit syntax errors, guaranteeing 100% valid JSON.
9. **Mobile OS Integration (The 6 Core Tools)**: Permission-controlled execution for Reminders (`AlarmManager`), Calendar (`CalendarProvider`), Notes (`Room` DB), Document Search, Local RAG, and Message Drafting.
10. **Zero-Trust Privacy Architecture**: Manifest omitting internet permissions, human-in-the-loop confirmation gates, prompt injection defense, and SQLite audit logging.
11. **Agent-Aware Quantization Benchmark (AAQB)**: Evaluating agents on Syntax Adherence, Tool Accuracy, Task Success Rate, TTFT, and battery consumption.
12. **Live Verification & Operational Sandbox**: Verified 100% offline execution in Airplane mode, local SQLite Room mirror, and passing automated test suite (`tests_agent/test_agent_system.py`).
13. **Workload Division & Engineering Roadmap**: Technical ownership for Faqre, Ekansh, and Guneet, leading to final Android APK deployment and university defense.
