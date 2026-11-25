"""
GGUF Model Manager & Quantization Registry.
Manages candidate SLMs, precision formats (Q8_0, Q4_K_M, Q4_0),
download pipelines, and edge memory profiling.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import os


class ModelPrecision(str, Enum):
    FP16 = "FP16"
    Q8_0 = "Q8_0"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"


@dataclass
class GGUFModelSpec:
    model_id: str
    architecture: str
    parameter_count_str: str
    precision: ModelPrecision
    repo_id: str
    filename: str
    estimated_size_mb: float
    estimated_ram_mb: float
    description: str


# -------------------------------------------------------------------------
# Candidate Model Catalog (GGUF universal edge format)
# -------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[ModelPrecision, GGUFModelSpec]] = {
    "llama-3.2-1b": {
        ModelPrecision.FP16: GGUFModelSpec(
            model_id="llama-3.2-1b",
            architecture="Llama-3.2-1B-Instruct",
            parameter_count_str="1.23B",
            precision=ModelPrecision.FP16,
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            filename="model.safetensors",
            estimated_size_mb=2460.0,
            estimated_ram_mb=3800.0,
            description="Uncompressed 16-bit baseline reference"
        ),
        ModelPrecision.Q8_0: GGUFModelSpec(
            model_id="llama-3.2-1b",
            architecture="Llama-3.2-1B-Instruct",
            parameter_count_str="1.23B",
            precision=ModelPrecision.Q8_0,
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
            estimated_size_mb=1320.0,
            estimated_ram_mb=1850.0,
            description="8-bit lossless integer quantization"
        ),
        ModelPrecision.Q4_K_M: GGUFModelSpec(
            model_id="llama-3.2-1b",
            architecture="Llama-3.2-1B-Instruct",
            parameter_count_str="1.23B",
            precision=ModelPrecision.Q4_K_M,
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            estimated_size_mb=804.0,
            estimated_ram_mb=1150.0,
            description="4-bit Medium K-Quant (Optimal Edge Sweet Spot)"
        ),
        ModelPrecision.Q4_0: GGUFModelSpec(
            model_id="llama-3.2-1b",
            architecture="Llama-3.2-1B-Instruct",
            parameter_count_str="1.23B",
            precision=ModelPrecision.Q4_0,
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_0.gguf",
            estimated_size_mb=747.0,
            estimated_ram_mb=980.0,
            description="4-bit Naive Symmetric Quantization"
        ),
    },
    "qwen2.5-0.5b": {
        ModelPrecision.FP16: GGUFModelSpec(
            model_id="qwen2.5-0.5b",
            architecture="Qwen2.5-0.5B-Instruct",
            parameter_count_str="490M",
            precision=ModelPrecision.FP16,
            repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            filename="model.safetensors",
            estimated_size_mb=980.0,
            estimated_ram_mb=1600.0,
            description="Uncompressed baseline reference"
        ),
        ModelPrecision.Q8_0: GGUFModelSpec(
            model_id="qwen2.5-0.5b",
            architecture="Qwen2.5-0.5B-Instruct",
            parameter_count_str="490M",
            precision=ModelPrecision.Q8_0,
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q8_0.gguf",
            estimated_size_mb=536.0,
            estimated_ram_mb=780.0,
            description="8-bit lossless reference"
        ),
        ModelPrecision.Q4_K_M: GGUFModelSpec(
            model_id="qwen2.5-0.5b",
            architecture="Qwen2.5-0.5B-Instruct",
            parameter_count_str="490M",
            precision=ModelPrecision.Q4_K_M,
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            estimated_size_mb=398.0,
            estimated_ram_mb=550.0,
            description="Ultra-compact 4-bit edge model"
        ),
        ModelPrecision.Q4_0: GGUFModelSpec(
            model_id="qwen2.5-0.5b",
            architecture="Qwen2.5-0.5B-Instruct",
            parameter_count_str="490M",
            precision=ModelPrecision.Q4_0,
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_0.gguf",
            estimated_size_mb=352.0,
            estimated_ram_mb=490.0,
            description="Ultra-compact 4-bit naive symmetric"
        ),
    }
}


class GGUFModelManager:
    """Manages GGUF model specifications, acquisition, and local caching."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_spec(self, model_id: str, precision: ModelPrecision) -> Optional[GGUFModelSpec]:
        """Retrieve model specification."""
        model_family = MODEL_SPECS.get(model_id.lower())
        if not model_family:
            return None
        return model_family.get(precision)

    def list_available_models(self) -> List[str]:
        """List all supported model IDs."""
        return list(MODEL_SPECS.keys())

    def download_model(
        self,
        model_id: str,
        precision: ModelPrecision,
        local_dir: Optional[str] = None
    ) -> str:
        """
        Download GGUF model weights from Hugging Face Hub.
        Returns the absolute local file path.
        """
        spec = self.get_spec(model_id, precision)
        if not spec:
            raise ValueError(f"Unknown model_id '{model_id}' or precision '{precision}'.")

        target_dir = local_dir or self.cache_dir
        expected_path = os.path.join(target_dir, spec.filename)

        if os.path.exists(expected_path):
            return expected_path

        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        return downloaded_path

    @staticmethod
    def calculate_compression_ratio(fp16_mb: float, quant_mb: float) -> float:
        """Calculate percentage reduction in memory footprint."""
        if fp16_mb <= 0:
            return 0.0
        return ((fp16_mb - quant_mb) / fp16_mb) * 100.0
