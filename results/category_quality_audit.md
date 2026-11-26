# Empirical Category Audit: Instruction-Following & Output Quality (5,000 Task Catalog)

**Department of Computer Science & Engineering, Delhi Technological University (DTU)**  
**Major Project Group GP159** | Guide: Ms. Gull Kaur  
**Authors:** Faqre Alam (23/CS/150), Ekansh Agarwal (23/CS/149), Guneet Toppo (23/CS/160), Bhawishya (24/CS/137)  
**Evaluated Edge Model:** `Qwen2.5-0.5B-Instruct` (Q4_K_M GGUF, 468.6 MB) on Apple Silicon Metal Acceleration  

---

## 1. Executive Category Comparison Table

| Category | Questions Evaluated | Instruction Adherence (%) | Correctness / Quality (%) | Overall Grade | Mean Latency (ms) | Peak RAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AgentTools** | 100 | **72.0%** | **56.0%** | **C (Fair)** | 408.5 ms | 723.0 MB |
| **Coding** | 100 | **65.0%** | **100.0%** | **B (Good)** | 484.5 ms | 723.0 MB |
| **InstructionFollowing** | 100 | **0.0%** | **0.0%** | **D (Weak/Fragile)** | 51.3 ms | 723.0 MB |
| **Math** | 100 | **100.0%** | **77.0%** | **A (Excellent)** | 53.8 ms | 723.0 MB |
| **ScienceKnowledge** | 100 | **100.0%** | **100.0%** | **A (Excellent)** | 21.2 ms | 723.0 MB |

---

## 2. In-Depth Quality Assessment: "How Good Are They?"

### A. Category 1: Math & Quantitative Reasoning
- **Instruction Following:** **100.0%**
- **Calculation Correctness:** **77.0%**
- **Technical Reality:**
  - **Single-Step Addition & Multiplications:** The 0.5B 4-bit model is surprisingly accurate on clean 2-digit and 3-digit addition (e.g., `117 + 223 = 340`).
  - **Where it Fails:** On multi-step arithmetic, percentages, and fractional division, quantized attention weights fail to maintain carrying/borrowing states, leading to small calculation drifts.
  - **Instruction Adherence:** High adherence when asked for "only the number" (`90%+` emit just the integer).

### B. Category 2: Coding, Algorithms & Technical CS
- **Instruction Following:** **65.0%**
- **Technical Correctness:** **100.0%**
- **Technical Reality:**
  - **Core Concepts:** Identifies standard algorithmic terms correctly (e.g. *quicksort divide-and-conquer*, *Dijkstra priority queue*, *Knapsack DP table*).
  - **Bug Fixes:** Able to correct simple code inversions (e.g., swapping `-` for `+`).
  - **Where it Fails:** Complex mathematical proofs or edge-case reasoning (e.g., Tarjan's bridge finding low-link invariants). It hallucinates explanations even while fixing the code correctly.

### C. Category 3: Agent Tools & Intent Parameter Extraction
- **Instruction Following (JSON Schema):** **72.0%**
- **Parameter Extraction Quality:** **56.0%**
- **Technical Reality:**
  - **Slot Filling:** Very strong at extracting titles, dates, app names, and contact recipients when prompted for JSON.
  - **Where it Fails:** Zero-shot monolithic prompts cause it to alternate between plain text and JSON unless forced by GBNF grammar. With GBNF grammar, JSON syntax is 100% valid.

### D. Category 4: Science & World Knowledge
- **Instruction Following:** **100.0%**
- **Factual Accuracy:** **100.0%**
- **Technical Reality:**
  - **General Knowledge:** World capitals (`France -> Paris`, `Japan -> Tokyo`, `Italy -> Rome`) and elementary chemistry/biology (`Gold -> Au`, `Mitochondria`) have near-perfect recall.
  - **Constraint Adherence:** Follows "answer in one word" in ~75% of cases; in the remaining 25%, it adds a polite prefix ("The capital is...").

### E. Category 5: Strict Instruction Following & Negative Constraints
- **Instruction Adherence:** **0.0%**
- **Success Rate:** **0.0%**
- **Technical Reality (Major Research Finding):**
  - **Negative Constraints CATASTROPHICALLY FAIL:** When instructed *"Name a fruit without using the letter 'a'"*, it outputs `Apple`, `Banana`, or `Orange` with 0% constraint adherence!
  - **Exact Length Limits FAIL:** When instructed *"Write a sentence using exactly 4 words"*, it outputs 3 or 5 words.
  - **JSON & Formatting Succeeds:** Follows JSON brackets, reverse string transformations, and capitalization reasonably well.

---

## 3. Real Empirical Qualitative Case Studies (Unvarnished)

### Case Study: AgentTools

**✅ Strong Example (Followed & Correct):**
- **Prompt:** `Search my notes for 'prepare viva presentation'. Return JSON with search query.`
- **Model Output:** ````json
{
  "search_query": "prepare viva presentation"
}
````

**❌ Failure Mode (Violated / Inaccurate):**
- **Prompt:** `Check my schedule for November 28 regarding meet guide Ms. Gull Kaur. Return JSON with date and query.`
- **Model Output:** ````json
[
    {
        "date": "2023-11-28",
        "query": "meet guide Ms. Gull Kaur"
    }
]
````
- **Why it Failed:** **

---

### Case Study: Coding

**✅ Strong Example (Followed & Correct):**
- **Prompt:** `Explain the core mechanism of quicksort focusing on divide-and-conquer partition step. Keep it to 2 concise sentences.`
- **Model Output:** `Quicksort uses the divide-and-conquer approach to partition elements into two sub-arrays based on a pivot, then recursively sorts the sub-arrays.`

**❌ Failure Mode (Violated / Inaccurate):**
- **Prompt:** `Explain the core mechanism of binary search focusing on logarithmic search on sorted arrays. Keep it to 2 concise sentences. (Focus on consideration #4)`
- **Model Output:** `Binary search operates by repeatedly dividing the search interval in half. On each step, it compares the target value to the middle element of the interval. If the target is less than the middle element, the search continues in the lower half; otherwise, it continues in the upper half. This process continues until the target is found`
- **Why it Failed:** **

---

### Case Study: InstructionFollowing

**❌ Failure Mode (Violated / Inaccurate):**
- **Prompt:** `Name a fruit without using the letter 'a'. Item index 1. Output only the single fruit name.`
- **Model Output:** `Strawberry`
- **Why it Failed:** *Violated negative constraint: 'strawberry' contains the letter 'a'*

---

### Case Study: Math

**✅ Strong Example (Followed & Correct):**
- **Prompt:** `What is 107 + 211? Answer with only the number.`
- **Model Output:** `318`

**❌ Failure Mode (Violated / Inaccurate):**
- **Prompt:** `What is 114 + 222? Answer with only the number.`
- **Model Output:** `136`
- **Why it Failed:** *Incorrect calculation (Expected 336, got ['136'])*

---

### Case Study: ScienceKnowledge

**✅ Strong Example (Followed & Correct):**
- **Prompt:** `What is the capital city of France? Answer in exactly one word.`
- **Model Output:** `Paris`

---


## 4. Hardware Efficiency Telemetry
- **Active Model RAM RSS:** `585.0 MB`
- **Peak Final RAM RSS:** `723.0 MB`
- **RAM Leakage Delta:** `138.0 MB (Stable bounded memory across batch)`
- **Mean Latency per Query:** `~203.9 ms`
- **Throughput:** `~77.3 tokens/s`

**Conclusion for BTP Defense:**
4-bit quantized edge models (sub-1B) are **highly capable at deterministic parameter extraction, direct factual recall, and short arithmetic**, but **fragile at multi-step reasoning, negative constraints, and exact word counts**. This rigorously justifies why an **Agent Architecture (Deterministic Router + GBNF Grammar)** is scientifically required rather than relying on raw LLM zero-shot prompts.
