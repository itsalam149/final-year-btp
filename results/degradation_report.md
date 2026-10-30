# Task-Conditioned Degradation Report (RQ1)

Degradation measured in percentage points (pp) against unquantized FP16 baseline.
Includes 95% paired bootstrap confidence intervals and relative drops.

| model        | method   |   bits |   delta_wikitext |   delta_arc_easy |   delta_hellaswag |   delta_winogrande |   delta_mmlu |   delta_gsm8k |   delta_ifeval |
|--------------|----------|--------|------------------|------------------|-------------------|--------------------|--------------|---------------|----------------|
| qwen2.5-0.5b | fp16     |     16 |              nan |                0 |               nan |                nan |          nan |           nan |            nan |
| qwen2.5-0.5b | fp16     |     16 |              nan |                5 |               nan |                nan |          nan |           nan |            nan |
