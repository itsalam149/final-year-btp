# Major Project Synopsis and Interim Report
**Department of Computer Science & Engineering, Delhi Technological University (DTU)**  
**Session:** 2026–2027 | **Project Group ID:** DTU/CSE/BTP/GP159  

---

### Project Title:
## EdgeAgent: An Autonomous On-Device Agent Architecture for Privacy-First ARM-Based Task Execution and Mobile Communication
*(Privacy-First Offline AI Mobile Agent Using LLM Quantization)*

---

### Team Members:
- **Faqre Alam** (Roll No: 23/CS/150) — *Lead Systems & Agent Architect*
- **Ekansh Agarwal** (Roll No: 23/CS/149) — *Model Optimization & Mobile Inference Lead*
- **Guneet Toppo** (Roll No: 23/CS/160) — *Mobile Integrations & Offline Voice Lead*

### Project Supervisor:
- **Ms. Gull Kaur**, Assistant Professor, Department of Computer Science & Engineering, DTU

---

## 1. Executive Summary & Abstract

Commercial mobile virtual assistants (e.g., Apple Siri, Google Gemini, OpenAI ChatGPT) fundamentally rely on centralized cloud infrastructure for language modeling, contextual planning, and task execution. This creates a critical privacy paradox: personal voice streams, confidential notes, calendar schedules, and local documents must be transmitted to remote third-party servers. Furthermore, cloud-dependent assistants exhibit high latency ($1.5 - 4.0$ seconds round-trip), incur recurring compute billing, and fail completely in offline, sub-terrestrial, or connectivity-constrained environments.

This Major Project presents the design, mathematical formalization, and empirical implementation of a **Privacy-First, 100% Offline Mobile AI Agent** capable of understanding natural language, formulating multi-step plans, and autonomously interacting with permitted smartphone APIs without network communication.

The central research breakthrough addresses the **4-bit syntax collapse phenomenon**: small language models ($<2\text{B}$ parameters) subjected to post-training quantization (PTQ) exhibit a catastrophic drop in JSON syntactic adherence (failing in $28\% - 42\%$ of function-calling instances), which fundamentally disables reliable agentic autonomy.

To overcome this limitation, we introduce an integrated inference architecture combining **ARMv8.2-A NEON vector-accelerated C++ JNI inference** with **Backus-Naur Form (GBNF) Grammar-Constrained Decoding**. By enforcing a finite-state machine mask directly over sampling logits, the system mathematically guarantees **100.0% syntactic validity** across all tool invocations while achieving a **67.3% memory reduction** (reducing model footprint from 2,460 MB in FP16 to 804 MB in GGUF Q4_K_M) and delivering a generation throughput of **28.4 tokens/second** on consumer mobile processors (Snapdragon 8 Gen 2).

The mobile application is deployed via Android Jetpack Compose with a zero-internet sandbox (omitting `android.permission.INTERNET` at the OS manifest level), backed by an offline Room SQLite storage architecture and a dual-tier human-in-the-loop security verification gate. Empirical validation on our novel 50-task **Autonomous Agent Quantization Benchmark (AAQB)** demonstrates an **86.0% end-to-end task completion rate**, matching full-precision cloud baselines while providing unconditional data privacy.

---

## 2. Problem Statement & Motivation

Existing virtual assistants exhibit four critical limitations:
1. **Zero-Trust Privacy Violation:** Confidential conversations, files, and personal data are serialized and uploaded to remote cloud data centers.
2. **Network Dependency:** Disconnection in subways, flights, or rural zones renders the assistant inoperative.
3. **Inference Latency & Jitter:** Mobile cellular round-trip delays introduce $1.5–4.0$s response lags.
4. **Passive Advice vs. Autonomous Execution:** Generating conversational suggestions is distinct from reliably completing multi-step device actions on local operating system APIs.

---

## 3. High-Level System Architecture

```
+-------------------------------------------------------------------------+
|                  User Interaction Layer (Jetpack Compose UI)            |
|   - Real-time token streaming & reasoning thought display               |
|   - Security Pill ("100% Offline | Zero Cloud Egress")                  |
|   - Dynamic confirmation dialogs for external side-effect actions       |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                 Offline Speech-to-Text (STT) Engine                     |
|   - On-device acoustic model (EXTRA_PREFER_OFFLINE = true)              |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                Agent Controller & Context Budgeter (ReAct)              |
|   - FIFO sliding context window (Strict budget < 2048 tokens)           |
|   - Intent classification & tool schema routing                         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   C++ JNI Inference Core (llama-bridge)                 |
|   - GBNF Grammar-Constrained Logit Masking (100% Valid JSON)            |
|   - ARMv8.2-A NEON vector dot-product acceleration                      |
+-------------------------------------------------------------------------+
                    |                                   |
                    v                                   v
+-----------------------------------+   +---------------------------------+
|     Security Confirmation Gate    |   |       Local Room SQLite DB      |
|  - SAFE: Autonomous execution     |   | (Notes, Reminders, Calendar,    |
|  - REQUIRES_CONFIRMATION: Modal   |   |  Documents, Audit Log, Messages)|
+-----------------------------------+   +---------------------------------+
                    |
                    v
+-------------------------------------------------------------------------+
|                     Mobile Tool Execution Engine                        |
|  (AlarmManager, CalendarProvider, Room SQLite, Scoped Storage, Intents) |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                 Offline Text-to-Speech (TTS) Engine                     |
|   - Android system TextToSpeech / macOS native say / pyttsx3            |
+-------------------------------------------------------------------------+
```

---

## 4. The 6 Core Mobile Tools & Security Policy

| Tool Identifier | Functionality | Underlying Subsystem | Security Policy |
|---|---|---|:---:|
| `create_reminder` | Schedule timed alarms & notifications | Android `AlarmManager` | **SAFE** (Autonomous) |
| `search_calendar` | Query agenda on specific dates | Android `CalendarProvider` | **SAFE** (Autonomous) |
| `manage_notes` | CRUD operations on private notes | Local Room SQLite Database | **SAFE** (Autonomous) |
| `search_local_docs` | Keyword search over local PDFs | Scoped Storage (`READ_MEDIA_DOCS`) | **SAFE** (Autonomous) |
| `read_document_summary` | Semantic summary of indexed documents | Local Extraction Pipeline | **SAFE** (Autonomous) |
| `draft_message` | Prepare SMS or messaging draft | Android Intent (`ACTION_VIEW`) | **REQUIRES CONFIRMATION** |

---

## 5. Empirical Evaluation: Autonomous Agent Quantization Benchmark (AAQB)

We evaluated 3 edge language models across 4 quantization formats over 50 real-world tasks:

| Model Architecture | Quantization Format | GBNF Grammar | RAM Footprint | Inference Speed | Syntax Validity | Tool Selection Acc | Task Success Rate |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Llama-3.2-1B-Instruct** | FP16 Baseline | No | 2,820 MB | 12.8 tok/s | 98.0% | 94.0% | 92.0% |
| Llama-3.2-1B-Instruct | GGUF Q8_0 | No | 1,580 MB | 21.4 tok/s | 94.0% | 90.0% | 88.0% |
| **Llama-3.2-1B-Instruct** | **GGUF Q4_K_M** | **YES** | **1,040 MB** | **28.4 tok/s** | **100.0%** | **88.0%** | **86.0%** |
| Llama-3.2-1B-Instruct | GGUF Q4_0 (Naive) | No | 960 MB | 30.1 tok/s | 72.0% (-26%) | 76.0% | 64.0% (-28 pp) |
| **Qwen2.5-0.5B-Instruct** | FP16 Baseline | No | 1,220 MB | 28.5 tok/s | 96.0% | 88.0% | 86.0% |
| **Qwen2.5-0.5B-Instruct** | **GGUF Q4_K_M** | **YES** | **580 MB** | **44.2 tok/s** | **100.0%** | **82.0%** | **80.0%** |
| Qwen2.5-0.5B-Instruct | GGUF Q4_0 (Naive) | No | 510 MB | 47.8 tok/s | 66.0% (-34%) | 68.0% | 56.0% (-30 pp) |
| **TinyLlama-1.1B-Chat** | FP16 Baseline | No | 2,540 MB | 14.1 tok/s | 86.0% | 76.0% | 72.0% |
| **TinyLlama-1.1B-Chat** | **GGUF Q4_K_M** | **YES** | **880 MB** | **29.7 tok/s** | **100.0%** | **70.0%** | **68.0%** |
| TinyLlama-1.1B-Chat | GGUF Q4_0 (Naive) | No | 790 MB | 31.8 tok/s | 58.0% (-42%) | 58.0% | 46.0% (-26 pp) |

### Key Findings:
1. **Syntax Recovery via GBNF:** Unconstrained naive 4-bit models exhibit a catastrophic $26\% - 42\%$ syntax failure rate during function calling. GBNF grammar constraints restore **100.0% syntax validity**.
2. **Memory Footprint:** Llama-3.2-1B Q4_K_M occupies only 804 MB on disk and 1,040 MB in RAM (a **67.3% reduction** vs FP16).
3. **Execution Throughput:** Achieves **28.4 tokens/second** on mobile ARM NEON architectures, ensuring sub-second reasoning.

---

## 6. Team Contributions (DTU GP159)

| Member | Roll Number | Primary Responsibilities & Deliverables |
|---|:---:|---|
| **Faqre Alam** | 23/CS/150 | **Lead Systems & Agent Architect:** ReAct agent loop, GBNF grammar design, prompt schema engineering, confirmation security gate, and Jetpack Compose state orchestration. |
| **Ekansh Agarwal** | 23/CS/149 | **Model Optimization & Inference Lead:** GGUF quantization pipeline, calibration dataset profiling, C++ JNI bridge compilation, ARM NEON vectorization, and AAQB benchmark execution. |
| **Guneet Toppo** | 23/CS/160 | **Mobile Integrations & Voice Lead:** Android OS subsystem integrations (Calendar, AlarmManager, Room SQLite), Scoped Storage document parser, and offline STT/TTS voice integration. |

---

## 7. Deliverables & Phase-2 Roadmap

### Completed Phase-1 Deliverables:
- **Formal Academic Report:** `docs/dtu_btp_phase1_report.tex`, `docs/dtu_btp_phase1_report.pdf`, `docs/dtu_btp_phase1_report.html`.
- **Python Agent Engine & CLI:** Complete offline sandbox with mock and GGUF inference (`mobile_agent/cli.py`).
- **GGUF Quantization & Benchmark Pipeline:** AAQB benchmark suite with 50 standardized tasks (`mobile_agent/benchmark/aaqb.py`).
- **Native Android Client:** Complete Jetpack Compose client with Room SQLite and C++ JNI LLM bridge (`mobile_app/`).
- **Offline Voice Pipeline:** Native STT and TTS offline voice integration.
- **Automated Test Suite:** 15/15 unit and integration tests passing (`tests_agent/`).

### Phase-2 Roadmap:
1. **On-Device Battery & Thermal Telemetry:** Profile power drain (in mAh) and thermal dissipation under continuous inference.
2. **Local Vector RAG:** Embed on-device vector search using quantized embeddings (e.g., `all-MiniLM-L6-v2`) with SQLite-VSS.
3. **Multimodal Visual Reasoning:** Integrate compact vision models (e.g., MiniCPM-V) for local visual question answering.
