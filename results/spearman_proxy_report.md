# Perplexity as a Proxy Reliability Report (RQ2)

Spearman Rank Correlation between delta_wikitext and downstream ΔAcc across tasks.
Evaluates whether perplexity is a faithful proxy or hides localized capability damage.

| task                               |   spearman_rho |   ci_low |   ci_high |   loo_min |   loo_max | influential_variant   |   p_value |   sample_size | proxy_status      |
|------------------------------------|----------------|----------|-----------|-----------|-----------|-----------------------|-----------|---------------|-------------------|
| arc_easy                           |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| hellaswag                          |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| winogrande                         |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| mmlu                               |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| gsm8k                              |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| ifeval                             |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |
| gsm8k_exact_match_flexible_extract |            nan |      nan |       nan |       nan |       nan |                       |       nan |             0 | INSUFFICIENT_DATA |

### Decision Criteria:
- **Faithful Proxy**: ρ <= -0.70 AND 95% CI upper bound < 0 (worsening PPL reliably tracks accuracy loss).
- **Decoupled**: |ρ| < 0.40 OR 95% CI spans 0 (PPL hides damage; task degradation decoupled from PPL).
- **Intermediate**: -0.70 < ρ <= -0.40.
