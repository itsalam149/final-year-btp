# Perplexity as a Proxy Reliability Report (RQ2)

Spearman Rank Correlation between delta_wikitext and downstream ΔAcc across tasks.
Evaluates whether perplexity is a faithful proxy or hides localized capability damage.

| task                               |   spearman_rho | ci_95          | loo_range      | influential_variant   |   p_value |   sample_size | proxy_status   |
|------------------------------------|----------------|----------------|----------------|-----------------------|-----------|---------------|----------------|
| arc_easy                           |        -0.9451 | [-1.00, -0.74] | [-0.97, -0.93] | bnb_nf4               |    0      |            15 | FAITHFUL_PROXY |
| hellaswag                          |        -0.9163 | [-1.00, -0.67] | [-0.95, -0.90] | bnb_nf4               |    0      |            15 | FAITHFUL_PROXY |
| winogrande                         |        -0.9198 | [-1.00, -0.66] | [-0.95, -0.90] | gptq                  |    0      |            15 | FAITHFUL_PROXY |
| mmlu                               |        -0.8407 | [-0.98, -0.46] | [-0.89, -0.80] | bnb_nf4               |    0.0001 |            15 | FAITHFUL_PROXY |
| gsm8k                              |        -0.6835 | [-0.93, -0.13] | [-0.75, -0.61] | fp16                  |    0.005  |            15 | INTERMEDIATE   |
| ifeval                             |        -0.7518 | [-0.94, -0.30] | [-0.83, -0.70] | gptq                  |    0.0012 |            15 | FAITHFUL_PROXY |
| gsm8k_exact_match_flexible_extract |        -0.6835 | [-0.93, -0.13] | [-0.75, -0.61] | fp16                  |    0.005  |            15 | INTERMEDIATE   |

### Decision Criteria:
- **Faithful Proxy**: ρ <= -0.70 AND 95% CI upper bound < 0 (worsening PPL reliably tracks accuracy loss).
- **Decoupled**: |ρ| < 0.40 OR 95% CI spans 0 (PPL hides damage; task degradation decoupled from PPL).
- **Intermediate**: -0.70 < ρ <= -0.40.
