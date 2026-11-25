"""
GGUF Quantization & Model Management Subsystem
Handles model acquisition, GGUF precision variants (Q8_0, Q4_K_M, Q4_0),
file integrity, and local caching for edge deployment.
"""

from mobile_agent.quantization.manager import GGUFModelManager, ModelPrecision, GGUFModelSpec

__all__ = ["GGUFModelManager", "ModelPrecision", "GGUFModelSpec"]
