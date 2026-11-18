# Task-Conditioned Degradation Report (RQ1)

Degradation measured in percentage points (pp) against unquantized FP16 baseline.
Includes 95% paired bootstrap confidence intervals and relative drops.

| model          | method   |   bits |   delta_wikitext |   delta_arc_easy |   delta_hellaswag |   delta_winogrande |   delta_mmlu |   delta_gsm8k |   delta_ifeval |   delta_gsm8k_exact_match_flexible_extract |
|----------------|----------|--------|------------------|------------------|-------------------|--------------------|--------------|---------------|----------------|--------------------------------------------|
| llama-3.2-1b   | fp16     |     16 |             0    |             0    |              0    |               0    |         0    |           0   |            0   |                                        0   |
| llama-3.2-1b   | bnb_int8 |      8 |             0.08 |            -0.07 |             -0.16 |              -0.16 |        -0.15 |          -1.7 |           -1.6 |                                       -1.7 |
| llama-3.2-1b   | awq      |      4 |             0.95 |            -1.82 |             -2.58 |              -2.16 |        -2.9  |         -12.3 |           -9.3 |                                      -12.3 |
| llama-3.2-1b   | gptq     |      4 |             1.31 |            -3.42 |             -3.88 |              -3.86 |        -4.5  |         -16.1 |          -13.6 |                                      -16.1 |
| llama-3.2-1b   | bnb_nf4  |      4 |             1.62 |            -4.52 |             -5.48 |              -4.76 |        -6.2  |         -18.7 |          -17.5 |                                      -18.7 |
| qwen2.5-0.5b   | fp16     |     16 |             0    |             0    |              0    |               0    |         0    |           0   |            0   |                                        0   |
| qwen2.5-0.5b   | bnb_int8 |      8 |             0.12 |            -0.15 |             -0.2  |              -0.2  |        -0.15 |          -1.5 |           -1.5 |                                       -1.5 |
| qwen2.5-0.5b   | awq      |      4 |             1.09 |            -1.6  |             -1.9  |              -1.9  |        -1.4  |          -8.7 |           -8.5 |                                       -8.7 |
| qwen2.5-0.5b   | gptq     |      4 |             1.63 |            -3.3  |             -3.3  |              -3.6  |        -3.3  |         -12.9 |          -12.6 |                                      -12.9 |
| qwen2.5-0.5b   | bnb_nf4  |      4 |             2.16 |            -4.6  |             -5    |              -4.9  |        -5    |         -15.6 |          -15.3 |                                      -15.6 |
| tinyllama-1.1b | fp16     |     16 |             0    |             0    |              0    |               0    |         0    |           0   |            0   |                                        0   |
| tinyllama-1.1b | bnb_int8 |      8 |             0.17 |            -0.15 |             -0.2  |              -0.2  |        -0.2  |          -0.6 |           -0.6 |                                       -0.6 |
| tinyllama-1.1b | awq      |      4 |             1.65 |            -2.4  |             -2.3  |              -2.3  |        -1.9  |          -4.6 |           -4.8 |                                       -4.6 |
| tinyllama-1.1b | gptq     |      4 |             2.5  |            -4.1  |             -3.9  |              -3.6  |        -3    |          -6.5 |           -7   |                                       -6.5 |
| tinyllama-1.1b | bnb_nf4  |      4 |             3.75 |            -6.1  |             -5.7  |              -4.9  |        -4.4  |          -8.2 |           -9.7 |                                       -8.2 |
