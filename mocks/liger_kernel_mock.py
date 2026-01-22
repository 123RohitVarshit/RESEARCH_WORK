"""
Liger Kernel Mock Module

Provides mock implementation of Liger Kernel to allow pedagogicalrl
to import without having liger-kernel installed.

Liger Kernel provides optimized CUDA kernels for transformer training.
Since we're using API-based inference, we don't need these optimizations.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


class LigerFusedLinearGRPOLoss:
    """
    Mock Liger fused linear GRPO loss class.
    
    In the real implementation, this provides an optimized CUDA kernel
    for computing GRPO loss. Our mock does nothing as we don't train locally.
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return 0.0


def setup_liger_kernel_mock() -> None:
    """
    Inject mock Liger Kernel modules into sys.modules.
    
    Liger Kernel is used for optimized loss computation during training.
    Since we use API inference, we don't need the actual kernels.
    """
    # Skip if already mocked
    if "liger_kernel" in sys.modules:
        return
    
    # Create main liger_kernel module
    liger_kernel = _create_mock_module("liger_kernel")
    
    # Create liger_kernel.chunked_loss submodule
    chunked_loss = _create_mock_module("liger_kernel.chunked_loss")
    chunked_loss.LigerFusedLinearGRPOLoss = LigerFusedLinearGRPOLoss
    
    liger_kernel.chunked_loss = chunked_loss
    
    # Register modules
    sys.modules["liger_kernel"] = liger_kernel
    sys.modules["liger_kernel.chunked_loss"] = chunked_loss


if __name__ == "__main__":
    setup_liger_kernel_mock()
    print("✅ Liger Kernel mock setup successful")
    
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
    loss = LigerFusedLinearGRPOLoss()
    print(f"   Created loss: {loss}")
