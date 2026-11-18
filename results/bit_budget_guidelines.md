# Practical Bit-Budget Guidelines (RQ3)

Evaluated under DTU Working Acceptance Rule: **ΔAcc loss <= 3.0 percentage points with 95% CI**.

| Task                               | Capability Category               |   Min Acceptable Bits | Confidence Tier                     | Best 4-Bit Format   |
|------------------------------------|-----------------------------------|-----------------------|-------------------------------------|---------------------|
| arc_easy                           | Factual Recall                    |                     4 | High (4-bit Acceptable with 95% CI) | awq (-1.60 pp)      |
| hellaswag                          | Commonsense Reasoning             |                     4 | High (4-bit Acceptable with 95% CI) | awq (-1.90 pp)      |
| winogrande                         | Coreference / Commonsense         |                     4 | High (4-bit Acceptable with 95% CI) | awq (-1.90 pp)      |
| mmlu                               | Academic / Domain Knowledge       |                     4 | High (4-bit Acceptable with 95% CI) | awq (-1.40 pp)      |
| gsm8k                              | Multi-Step Mathematical Reasoning |                     8 | High (8-bit Required, 4-bit Fails)  | awq (-4.60 pp)      |
| ifeval                             | Instruction Following             |                     8 | High (8-bit Required, 4-bit Fails)  | awq (-4.80 pp)      |
| gsm8k_exact_match_flexible_extract | General                           |                     8 | High (8-bit Required, 4-bit Fails)  | awq (-4.60 pp)      |

## Sensitivity Analysis Across Thresholds

|   threshold_pp |   total_tested |   acceptable_count |   acceptable_pct |   marginal_count |   rejected_count |
|----------------|----------------|--------------------|------------------|------------------|------------------|
|              1 |             84 |                  0 |              0   |               15 |               69 |
|              2 |             84 |                 15 |             17.9 |               12 |               57 |
|              3 |             84 |                 31 |             36.9 |                3 |               50 |
|              5 |             84 |                 56 |             66.7 |                0 |               28 |
