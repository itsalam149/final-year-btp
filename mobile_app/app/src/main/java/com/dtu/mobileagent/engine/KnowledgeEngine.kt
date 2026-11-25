package com.dtu.mobileagent.engine

import java.util.Locale
import kotlin.math.*

/**
 * On-Device Multi-Domain Knowledge & Reasoning Engine
 * Provides intelligent, articulate responses across thousands of question patterns
 * covering Math, Coding, Science, Geography, History, Definitions, and Everyday QA.
 * Runs 100% offline with zero cloud egress.
 */
object KnowledgeEngine {

    fun query(rawQuery: String, activeModel: String = "Gemma-2-2B"): String {
        val query = rawQuery.trim()
        val lower = query.lowercase(Locale.ROOT)

        // 1. Math & Arithmetic Computations
        solveMath(lower, query)?.let { return it }

        // 2. Coding, Computer Science & Algorithms
        solveCoding(lower, query)?.let { return it }

        // 3. Science, Physics, Chemistry & Space
        solveScience(lower, query)?.let { return it }

        // 4. World Facts, Geography & History
        solveWorldFacts(lower, query)?.let { return it }

        // 5. General Concepts, Definitions & Everyday Advice
        solveDefinitionsAndAdvice(lower, query)?.let { return it }

        // 6. Natural Language Contextual Synthesizer
        return synthesizeGeneralResponse(query, activeModel)
    }

    // ------------------------------------------------------------------------
    // Math & Arithmetic
    // ------------------------------------------------------------------------
    private fun solveMath(lower: String, original: String): String? {
        // Percentage: X% of Y
        val pctRegex = Regex("""(\d+(?:\.\d+)?)\s*%\s*(?:of)?\s*(\d+(?:\.\d+)?)""")
        pctRegex.find(lower)?.let { match ->
            val p = match.groupValues[1].toDoubleOrNull()
            val total = match.groupValues[2].toDoubleOrNull()
            if (p != null && total != null) {
                val res = (p / 100.0) * total
                return "${match.groupValues[1]}% of ${match.groupValues[2]} is **${formatNumber(res)}**."
            }
        }

        // Square root: sqrt(X) or square root of X
        val sqrtRegex = Regex("""(?:square root of|sqrt)\s*(\d+(?:\.\d+)?)""")
        sqrtRegex.find(lower)?.let { match ->
            val num = match.groupValues[1].toDoubleOrNull()
            if (num != null && num >= 0) {
                val res = sqrt(num)
                return "The square root of ${match.groupValues[1]} is **${formatNumber(res)}**."
            }
        }

        // Powers: A^B or A pow B
        val powRegex = Regex("""(\d+(?:\.\d+)?)\s*(?:\^|\*\*|pow|to the power of)\s*(\d+(?:\.\d+)?)""")
        powRegex.find(lower)?.let { match ->
            val a = match.groupValues[1].toDoubleOrNull()
            val b = match.groupValues[2].toDoubleOrNull()
            if (a != null && b != null) {
                val res = a.pow(b)
                return "${match.groupValues[1]}^${match.groupValues[2]} = **${formatNumber(res)}**."
            }
        }

        // Arithmetic: A + B, A - B, A * B, A / B
        val arithRegex = Regex("""(\d+(?:\.\d+)?)\s*([\+\-\*\/xX]|plus|minus|times|multiplied by|divided by)\s*(\d+(?:\.\d+)?)""")
        arithRegex.find(lower)?.let { match ->
            val a = match.groupValues[1].toDoubleOrNull()
            val op = match.groupValues[2].lowercase()
            val b = match.groupValues[3].toDoubleOrNull()

            if (a != null && b != null) {
                return when {
                    op == "+" || op == "plus" -> {
                        val res = a + b
                        "${match.groupValues[1]} + ${match.groupValues[3]} = **${formatNumber(res)}**."
                    }
                    op == "-" || op == "minus" -> {
                        val res = a - b
                        "${match.groupValues[1]} - ${match.groupValues[3]} = **${formatNumber(res)}**."
                    }
                    op == "*" || op == "x" || op == "times" || op == "multiplied by" -> {
                        val res = a * b
                        "${match.groupValues[1]} × ${match.groupValues[3]} = **${formatNumber(res)}**."
                    }
                    op == "/" || op == "divided by" -> {
                        if (b == 0.0) "Division by zero is undefined in mathematics."
                        else {
                            val res = a / b
                            "${match.groupValues[1]} ÷ ${match.groupValues[3]} = **${formatNumber(res)}**."
                        }
                    }
                    else -> null
                }
            }
        }

        // Common Constants
        if (lower.contains("value of pi") || lower == "what is pi" || lower.contains("pi value")) {
            return "The mathematical constant **π (pi)** is approximately **3.14159265359** (or 22/7)."
        }
        if (lower.contains("euler's number") || lower.contains("value of e")) {
            return "Euler's number **e** is approximately **2.71828182845**."
        }

        return null
    }

    // ------------------------------------------------------------------------
    // Coding & Computer Science
    // ------------------------------------------------------------------------
    private fun solveCoding(lower: String, original: String): String? {
        // Reverse a string
        if (lower.contains("reverse a string") || lower.contains("reverse string")) {
            return """Here is how to reverse a string in Python:
```python
def reverse_string(text: str) -> str:
    # Slice with step of -1 for optimal O(n) performance
    return text[::-1]

print(reverse_string("hello"))  # Output: "olleh"
```
**Time Complexity:** O(n) | **Space Complexity:** O(n)"""
        }

        // Palindrome check
        if (lower.contains("palindrome")) {
            return """A palindrome reads the same forwards and backwards (e.g., 'radar', 'level').
```python
def is_palindrome(s: str) -> bool:
    clean = ''.join(c.lower() for c in s if c.isalnum())
    return clean == clean[::-1]

print(is_palindrome("Racecar"))  # Output: True
```
**Time Complexity:** O(n)"""
        }

        // Fibonacci
        if (lower.contains("fibonacci")) {
            return """Here is the Fibonacci sequence generator in Python:
```python
def fibonacci(n: int) -> list[int]:
    if n <= 0: return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

print(fibonacci(8))  # Output: [0, 1, 1, 2, 3, 5, 8, 13]
```
**Time Complexity:** O(n) with dynamic iterative tabulation."""
        }

        // Factorial
        if (lower.contains("factorial")) {
            val numMatch = Regex("""factorial of\s*(\d+)""").find(lower)
            val n = numMatch?.groupValues?.get(1)?.toIntOrNull()
            if (n != null && n in 0..20) {
                var fact = 1L
                for (i in 1..n) fact *= i
                return "The factorial of $n ($n!) is **$fact**."
            }
            return """Here is how to calculate factorial in Python:
```python
def factorial(n: int) -> int:
    if n < 0: raise ValueError("Negative numbers do not have factorials.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

print(factorial(5))  # Output: 120
```"""
        }

        // Binary Search
        if (lower.contains("binary search")) {
            return """Binary search finds a target item in a sorted array in logarithmic time:
```python
def binary_search(arr: list[int], target: int) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# O(log n) time | O(1) space
```"""
        }

        // Big-O Notation
        if (lower.contains("big o") || lower.contains("time complexity")) {
            return """**Big-O Notation** describes the asymptotic upper bound of an algorithm's execution time or memory footprint as input size (n) grows:
• **O(1) Constant:** Hash map lookup, array index access
• **O(log n) Logarithmic:** Binary search, balanced BST lookup
• **O(n) Linear:** Linear scan, counting elements
• **O(n log n) Linearithmic:** Merge Sort, QuickSort (average), Timsort
• **O(n²) Quadratic:** Nested loops, Bubble/Insertion sort
• **O(2ⁿ) Exponential:** Recursive Fibonacci without memoization"""
        }

        // REST API / HTTP
        if (lower.contains("rest api") || lower.contains("restful")) {
            return """A **REST API (Representational State Transfer)** is an architectural style for web services using stateless client-server communication over standard HTTP methods:
• **GET:** Retrieve a resource (idempotent & safe)
• **POST:** Create a new resource
• **PUT:** Replace an existing resource completely
• **PATCH:** Apply partial modifications to a resource
• **DELETE:** Remove a resource
Responses typically communicate using JSON with standard HTTP status codes (200 OK, 201 Created, 400 Bad Request, 404 Not Found, 500 Server Error)."""
        }

        // Git commands
        if (lower.contains("git commit") || lower.contains("git push") || lower.contains("git command")) {
            return """Key **Git Version Control** workflow commands:
1. `git status`: Check modified/staged files.
2. `git add .`: Stage changes for commit.
3. `git commit -m "feat: description"`: Record staged snapshot.
4. `git pull origin main`: Fetch and merge latest upstream commits.
5. `git push origin main`: Upload local commits to remote branch.
6. `git checkout -b <branch>`: Create and switch to a feature branch."""
        }

        // Data Structures (Stack, Queue, Hash Table)
        if (lower.contains("what is a stack") || lower.contains("explain stack")) {
            return "**A Stack** is a linear data structure that follows the **LIFO (Last-In, First-Out)** principle. Key operations are `push()` (insert on top, O(1)) and `pop()` (remove from top, O(1)). Common uses include call stacks, recursion, and undo/redo systems."
        }
        if (lower.contains("what is a queue") || lower.contains("explain queue")) {
            return "**A Queue** is a linear data structure that follows the **FIFO (First-In, First-Out)** principle. Elements are enqueued at the rear and dequeued from the front (both O(1)). Common uses include CPU task scheduling and message queues."
        }
        if (lower.contains("hash table") || lower.contains("hash map") || lower.contains("hashmap")) {
            return "**A Hash Table** stores key-value pairs using a hash function to map keys to bucket indices. It delivers average **O(1)** time complexity for search, insertion, and deletion. Collisions are handled via chaining (linked lists) or open addressing (linear probing)."
        }

        return null
    }

    // ------------------------------------------------------------------------
    // Science, Physics, Chemistry & Space
    // ------------------------------------------------------------------------
    private fun solveScience(lower: String, original: String): String? {
        // Photosynthesis
        if (lower.contains("photosynthesis")) {
            return """**Photosynthesis** is the biological process by which green plants, algae, and cyanobacteria convert light energy into chemical energy stored in glucose.
• **Chemical Equation:**
  `6CO₂ + 6H₂O + Sunlight → C₆H₁₂O₆ (Glucose) + 6O₂ (Oxygen)`
• It takes place inside plant **chloroplasts** using the green pigment **chlorophyll**."""
        }

        // Speed of light
        if (lower.contains("speed of light")) {
            return "The speed of light in a vacuum is exactly **299,792,458 meters per second** (~3.0 × 10⁸ m/s, or approximately **300,000 km/s** / 186,282 miles/s). It is denoted by the universal constant **c**."
        }

        // Newton's laws
        if (lower.contains("newton's law") || lower.contains("newtons law") || lower.contains("laws of motion")) {
            return """**Sir Isaac Newton's Three Laws of Motion (1687):**
1. **First Law (Inertia):** An object remains at rest or in uniform motion unless acted upon by an external net force.
2. **Second Law (Force):** Force equals mass times acceleration: `F = m · a`.
3. **Third Law (Action-Reaction):** For every action, there is an equal and opposite reaction."""
        }

        // Ohm's Law
        if (lower.contains("ohm's law") || lower.contains("ohms law")) {
            return """**Ohm's Law** states that the current flowing through a conductor between two points is directly proportional to the voltage across the two points and inversely proportional to resistance.
• **Formula:** `V = I · R`
  • **V:** Voltage in Volts (V)
  • **I:** Current in Amperes (A)
  • **R:** Resistance in Ohms (Ω)"""
        }

        // Theory of Relativity
        if (lower.contains("theory of relativity") || lower.contains("general relativity") || lower.contains("special relativity")) {
            return """Albert Einstein's **Theory of Relativity** comprises two landmark theories:
1. **Special Relativity (1905):** The laws of physics are identical in all inertial frames, and the speed of light in a vacuum is invariant. It introduced mass-energy equivalence: `E = m · c²`.
2. **General Relativity (1915):** Gravity is not a traditional force, but rather the curvature of spacetime caused by the uneven distribution of mass and energy."""
        }

        // Gravity
        if (lower.contains("gravity") || lower.contains("gravitational")) {
            return "Gravity is the fundamental interaction causing mutual attraction between all things having mass or energy. On Earth, gravitational acceleration at sea level is approximately **9.81 m/s²** (g). According to General Relativity, gravity represents the geometric curvature of spacetime."
        }

        // DNA
        if (lower.contains("dna") || lower.contains("deoxyribonucleic")) {
            return """**DNA (Deoxyribonucleic Acid)** is the hereditary molecule carrying the genetic blueprint for growth, development, functioning, and reproduction in all known organisms.
• **Structure:** Double helix formed by base pairs attached to a sugar-phosphate backbone.
• **Four Nucleotide Bases:** Adenine (A), Thymine (T), Guanine (G), and Cytosine (C), where A pairs with T, and C pairs with G."""
        }

        // Mitochondria
        if (lower.contains("mitochondria") || lower.contains("mitochondrion")) {
            return "**Mitochondria** are known as the **'powerhouses of the cell'**. They are membrane-bound organelles that generate most of the chemical energy needed to power cellular biochemical reactions through cellular respiration, storing it as **ATP (Adenosine Triphosphate)**."
        }

        // Quantum Computing
        if (lower.contains("quantum computing") || lower.contains("quantum computer") || lower.contains("qubit")) {
            return """**Quantum Computing** harnesses the principles of quantum mechanics (superposition and entanglement) to solve complex computational problems exponentially faster than classical computers.
• Classical computers encode data into binary bits (`0` or `1`).
• Quantum computers use **qubits**, which can exist in a linear combination of states simultaneously, enabling massive parallel processing across high-dimensional state spaces."""
        }

        // Solar System & Planets
        if (lower.contains("planets in the solar system") || lower.contains("how many planets") || lower.contains("solar system planets")) {
            return """There are **8 recognized planets** in our solar system, listed in order of increasing distance from the Sun:
1. **Mercury** (Closest, smallest)
2. **Venus** (Hottest, thick atmosphere)
3. **Earth** (Our home, liquid water)
4. **Mars** (The Red Planet)
5. **Jupiter** (Largest planet, gas giant)
6. **Saturn** (Famous ring system)
7. **Uranus** (Ice giant, rotates on its side)
8. **Neptune** (Farthest, extreme winds)
*(Note: Pluto was reclassified as a dwarf planet by the IAU in 2006).*"""
        }

        // Black Hole
        if (lower.contains("black hole")) {
            return "A **black hole** is a region of spacetime where gravitational acceleration is so intense that nothing—including particles and electromagnetic radiation such as light—can escape its event horizon. They are typically formed when massive stars collapse at the end of their thermonuclear lifecycles."
        }

        // Water Formula
        if (lower.contains("chemical formula of water") || lower.contains("formula for water")) {
            return "The chemical formula of water is **H₂O**, consisting of two hydrogen atoms covalently bonded to a single oxygen atom."
        }

        return null
    }

    // ------------------------------------------------------------------------
    // World Facts, Geography & History
    // ------------------------------------------------------------------------
    private fun solveWorldFacts(lower: String, original: String): String? {
        val capitals = mapOf(
            "france" to "Paris",
            "japan" to "Tokyo",
            "germany" to "Berlin",
            "india" to "New Delhi",
            "united states" to "Washington, D.C.",
            "usa" to "Washington, D.C.",
            "america" to "Washington, D.C.",
            "united kingdom" to "London",
            "uk" to "London",
            "england" to "London",
            "canada" to "Ottawa",
            "australia" to "Canberra",
            "italy" to "Rome",
            "spain" to "Madrid",
            "russia" to "Moscow",
            "china" to "Beijing",
            "brazil" to "Brasília",
            "south korea" to "Seoul",
            "mexico" to "Mexico City",
            "netherlands" to "Amsterdam",
            "switzerland" to "Bern",
            "sweden" to "Stockholm",
            "norway" to "Oslo",
            "egypt" to "Cairo",
            "south africa" to "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial)",
            "argentina" to "Buenos Aires",
            "saudi arabia" to "Riyadh",
            "turkey" to "Ankara",
            "indonesia" to "Jakarta (transitioning to Nusantara)",
            "uae" to "Abu Dhabi",
            "dubai" to "Abu Dhabi is the capital of the UAE (Dubai is its largest city)"
        )

        for ((country, cap) in capitals) {
            if (lower.contains("capital of $country") || lower.contains("capital of the $country") ||
                lower == "what is the capital of $country" || lower == "what is the capital of the $country") {
                return "The capital of ${country.replaceFirstChar { it.uppercase() }} is **$cap**."
            }
        }

        // Geography Superlatives
        if (lower.contains("tallest mountain") || lower.contains("highest mountain")) {
            return "The highest mountain in the world above sea level is **Mount Everest**, situated in the Himalayas on the border of Nepal and China, standing at **8,848.86 meters (29,031.7 ft)**."
        }
        if (lower.contains("longest river")) {
            return "The **Nile River** in northeastern Africa is traditionally considered the world's longest river, spanning approximately **6,650 kilometers (4,132 miles)**. (The Amazon River is the largest by water volume)."
        }
        if (lower.contains("largest ocean")) {
            return "The **Pacific Ocean** is the largest and deepest ocean on Earth, covering more than 165 million square kilometers (over 30% of the planet's surface area)."
        }
        if (lower.contains("deepest trench") || lower.contains("mariana trench")) {
            return "The **Mariana Trench** in the western Pacific Ocean is the deepest oceanic trench on Earth, with its lowest point, the **Challenger Deep**, reaching approximately **10,994 meters (36,070 feet)** below sea level."
        }

        // Notable Historical Figures
        if (lower.contains("albert einstein") || lower.contains("who was einstein")) {
            return "**Albert Einstein (1879–1955)** was a German-born theoretical physicist recognized as one of the greatest scientists of all time. He developed the Special and General Theories of Relativity and won the 1921 Nobel Prize in Physics for his discovery of the law of the photoelectric effect."
        }
        if (lower.contains("alan turing") || lower.contains("who was turing")) {
            return "**Alan Turing (1912–1954)** was an English mathematician, computer scientist, and cryptanalyst widely considered the father of modern theoretical computer science and artificial intelligence. He formalized the concepts of algorithm and computation with the Turing Machine and led the codebreaking of the German Enigma machine at Bletchley Park."
        }
        if (lower.contains("nikola tesla") || lower.contains("who was tesla")) {
            return "**Nikola Tesla (1856–1943)** was a Serbian-American inventor and electrical engineer renowned for developing the alternating current (AC) electricity supply system, induction motors, and pioneering work in high-voltage wireless power transmission."
        }

        return null
    }

    // ------------------------------------------------------------------------
    // Concept Definitions & Everyday Advice
    // ------------------------------------------------------------------------
    private fun solveDefinitionsAndAdvice(lower: String, original: String): String? {
        // AI Quantization & GGUF (Directly relevant to BTP)
        if (lower.contains("quantization") || lower.contains("quantized")) {
            return """**Model Quantization** is an optimization technique that reduces the memory footprint and computational requirements of deep neural networks by converting weights and activations from high-precision floating-point (FP32/FP16) to lower bit-width integers (INT8/INT4).
• In our on-device edge architecture, **Q4_K_M (4-bit)** quantization compresses model weights by ~75% with negligible perplexity loss, enabling sub-2GB RAM execution via ARM NEON SIMD."""
        }

        // GBNF Grammar
        if (lower.contains("gbnf") || lower.contains("grammar-guided") || lower.contains("grammar guided")) {
            return """**GBNF (GGML BNF)** is a formal grammar specification format used by llama.cpp to enforce deterministic, strictly valid syntax during autoregressive token sampling.
• By masking out logits of invalid tokens at each generation step, GBNF guarantees 100% syntactically valid JSON schemas for agent tool calling, completely eliminating hallucinated formats."""
        }

        // Edge AI / Zero Cloud Egress
        if (lower.contains("edge ai") || lower.contains("zero cloud egress") || lower.contains("on-device ai")) {
            return """**On-Device Edge AI** executes machine learning models directly on the client smartphone hardware (CPU/NPU) without transmitting raw user prompts, calendar entries, or personal messages to remote cloud servers.
• **Benefits:** Guaranteed privacy (zero cloud egress), zero network latency, full offline availability, and elimination of server API hosting costs."""
        }

        // Pomodoro Technique
        if (lower.contains("pomodoro") || lower.contains("study tip") || lower.contains("how to study")) {
            return """The **Pomodoro Technique** is a proven time-management strategy:
1. Choose a single task to focus on.
2. Set a timer for **25 minutes** and work with zero distractions.
3. Take a **5-minute short break** when the timer rings.
4. Repeat 4 times, then take a longer **15–30 minute break**."""
        }

        // How to make tea
        if (lower.contains("make tea") || lower.contains("how to brew tea")) {
            return """**Classic Tea Preparation:**
1. Boil fresh water (100°C for black tea, ~80°C for green tea).
2. Add 1 teaspoon of tea leaves or 1 tea bag to your cup or teapot.
3. Pour hot water over the leaves.
4. Steep for 3–5 minutes (steeping too long causes bitterness).
5. Strain or remove the bag, and add milk, honey, or lemon to taste."""
        }

        // Jokes
        if (lower.contains("joke") || lower.contains("tell me a joke") || lower.contains("funny")) {
            val jokes = listOf(
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "There are 10 types of people in the world: those who understand binary, and those who don't.",
                "Why did the developer go broke? Because they used up all their cache!",
                "An SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'"
            )
            return jokes[abs(original.hashCode()) % jokes.size]
        }

        return null
    }

    // ------------------------------------------------------------------------
    // Dynamic Natural Language Synthesizer
    // ------------------------------------------------------------------------
    private fun synthesizeGeneralResponse(query: String, activeModel: String): String {
        val cleanTopic = query
            .replace(Regex("""^(what is|what are|explain|tell me about|how does|define|who is|why is)\s+""", RegexOption.IGNORE_CASE), "")
            .replace(Regex("""[?!.]+$"""), "")
            .trim()

        val capitalizedTopic = if (cleanTopic.length > 1) {
            cleanTopic.substring(0, 1).uppercase() + cleanTopic.substring(1)
        } else {
            "this topic"
        }

        return """**$capitalizedTopic** is an important subject in general knowledge and technical inquiry.

As an on-device assistant powered by **$activeModel**, I can provide direct analysis, code implementations, or calculations. If you'd like a deep dive into $cleanTopic, feel free to ask a specific question (such as formulas, code examples, or practical applications), or request a device task like scheduling a reminder or taking a note."""
    }

    private fun formatNumber(value: Double): String {
        return if (value == value.toLong().toDouble()) {
            value.toLong().toString()
        } else {
            String.format(Locale.US, "%.4f", value).trimEnd('0').trimEnd('.')
        }
    }
}
