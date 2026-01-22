"""
vLLM Mock Module

Provides mock implementations of vLLM classes and functions to allow
pedagogicalrl to import without having vLLM installed.

vLLM is a high-performance LLM inference library that requires CUDA.
This mock allows the code to run using API-based inference instead.
"""

import sys
import types
from dataclasses import dataclass
from typing import List, Any, Optional
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


# =============================================================================
# Mock Data Classes (matching vLLM's interface)
# =============================================================================

@dataclass
class SamplingParams:
    """Mock vLLM SamplingParams for text generation configuration."""
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 100
    n: int = 1
    logits_processors: Any = None
    stop: Optional[List[str]] = None


@dataclass
class CompletionOutput:
    """Mock vLLM CompletionOutput representing a single completion."""
    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float
    logprobs: List[Any]


@dataclass
class RequestOutput:
    """Mock vLLM RequestOutput containing all completions for a request."""
    request_id: str
    prompt: str
    outputs: List[CompletionOutput]
    prompt_token_ids: List[int]
    prompt_logprobs: List[Any]
    finished: bool


class PoolingOutput:
    """Mock vLLM PoolingOutput for embedding operations."""
    pass


class LLM:
    """
    Mock vLLM LLM class.
    
    In the real vLLM, this loads model weights and runs inference.
    This mock does nothing - actual inference is handled via API.
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept any arguments to match vLLM's flexible constructor."""
        pass
    
    def encode(self, *args: Any, **kwargs: Any) -> List:
        """Mock encode method."""
        return []
    
    def chat(self, *args: Any, **kwargs: Any) -> List:
        """Mock chat method."""
        return []
    
    def generate(self, *args: Any, **kwargs: Any) -> List:
        """Mock generate method."""
        return []


class PoolerConfig:
    """Mock vLLM PoolerConfig for embedding model configuration."""
    
    def __init__(self, pooling_type: str, **kwargs: Any) -> None:
        self.pooling_type = pooling_type


# =============================================================================
# Mock Functions for Distributed Operations
# =============================================================================

def destroy_model_parallel() -> None:
    """Mock function to destroy model parallel groups."""
    pass


def destroy_distributed_environment() -> None:
    """Mock function to destroy distributed environment."""
    pass


# =============================================================================
# Setup Function
# =============================================================================

def setup_vllm_mock() -> None:
    """
    Inject mock vLLM modules into sys.modules.
    
    This tricks Python into believing vLLM is installed, allowing
    pedagogicalrl's import statements to succeed.
    """
    # Skip if already mocked
    if "vllm" in sys.modules and hasattr(sys.modules["vllm"], "_is_mock"):
        return
    
    # Create main vllm module
    vllm = _create_mock_module("vllm")
    vllm._is_mock = True
    
    # Attach classes to vllm module
    vllm.SamplingParams = SamplingParams
    vllm.CompletionOutput = CompletionOutput
    vllm.RequestOutput = RequestOutput
    vllm.PoolingOutput = PoolingOutput
    vllm.LLM = LLM
    
    # Create vllm.config submodule
    vllm_config = _create_mock_module("vllm.config")
    vllm_config.PoolerConfig = PoolerConfig
    vllm.config = vllm_config
    
    # Create vllm.distributed submodule
    vllm_distributed = _create_mock_module("vllm.distributed")
    
    # Create vllm.distributed.parallel_state submodule
    vllm_parallel_state = _create_mock_module("vllm.distributed.parallel_state")
    vllm_parallel_state.destroy_model_parallel = destroy_model_parallel
    vllm_parallel_state.destroy_distributed_environment = destroy_distributed_environment
    
    vllm_distributed.parallel_state = vllm_parallel_state
    vllm.distributed = vllm_distributed
    
    # Register all modules
    sys.modules["vllm"] = vllm
    sys.modules["vllm.config"] = vllm_config
    sys.modules["vllm.distributed"] = vllm_distributed
    sys.modules["vllm.distributed.parallel_state"] = vllm_parallel_state


if __name__ == "__main__":
    # Test the mock
    setup_vllm_mock()
    print("✅ vLLM mock setup successful")
    
    # Verify imports work
    import vllm
    params = vllm.SamplingParams(temperature=0.5)
    print(f"   Created SamplingParams: {params}")
