import torch
from transformers import AutoModelForCausalLM

print("Test 2: Load to CPU, then move to MPS")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print("Loaded to CPU successfully")
    
    if torch.backends.mps.is_available():
        model = model.to("mps")
        print("Moved to MPS successfully")
    print("Test 2 success")
except Exception as e:
    print(f"Test 2 failed: {e}")
