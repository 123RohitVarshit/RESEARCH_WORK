"""
Mock Libraries Package

This package provides mock implementations of heavy GPU libraries
(vLLM, DeepSpeed, Liger Kernel, PyTorch, pynvml, transformers) and pedagogicalrl internals
that are required by pedagogicalrl but not needed for API-based inference.

Usage:
    from mocks import setup_all_mocks
    setup_all_mocks()  # Call BEFORE importing pedagogicalrl
"""

from mocks.vllm_mock import setup_vllm_mock
from mocks.deepspeed_mock import setup_deepspeed_mock
from mocks.liger_kernel_mock import setup_liger_kernel_mock
from mocks.torch_mock import setup_torch_mock
from mocks.pynvml_mock import setup_pynvml_mock
from mocks.transformers_mock import setup_transformers_mock
from mocks.pedagogicalrl_mock import setup_pedagogicalrl_mocks


def setup_all_mocks() -> None:
    """
    Set up all mock libraries in sys.modules.
    
    This must be called BEFORE importing any pedagogicalrl modules,
    as Python caches imports and will fail if the real libraries
    are not installed.
    
    Example:
        from mocks import setup_all_mocks
        setup_all_mocks()
        
        # Now safe to import pedagogicalrl
        from pedagogicalrl.src.classroom import Classroom
    """
    setup_torch_mock()  # Must be first as others may depend on it
    setup_transformers_mock()  # HuggingFace transformers mock
    setup_vllm_mock()
    setup_deepspeed_mock()
    setup_liger_kernel_mock()
    setup_pynvml_mock()  # NVIDIA GPU monitoring mock
    setup_pedagogicalrl_mocks()  # Mock internal pedagogicalrl modules
    print("[OK] All mock libraries initialized successfully")


__all__ = [
    'setup_all_mocks',
    'setup_vllm_mock', 
    'setup_deepspeed_mock',
    'setup_liger_kernel_mock',
    'setup_torch_mock',
    'setup_pynvml_mock',
    'setup_transformers_mock',
    'setup_pedagogicalrl_mocks'
]
