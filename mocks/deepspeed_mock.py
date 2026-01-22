"""
DeepSpeed Mock Module

Provides mock implementation of DeepSpeed to allow pedagogicalrl
to import without having DeepSpeed installed.

DeepSpeed is Microsoft's deep learning optimization library for
distributed training. This mock allows the code to run without it.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


def setup_deepspeed_mock() -> None:
    """
    Inject mock DeepSpeed module into sys.modules.
    
    DeepSpeed is used for distributed training optimization.
    Since we're using API-based inference, we don't need it.
    """
    # Skip if already mocked
    if "deepspeed" in sys.modules:
        return
    
    # Create main deepspeed module
    deepspeed = _create_mock_module("deepspeed")
    
    # Add commonly accessed attributes
    deepspeed.init_distributed = lambda *args, **kwargs: None
    deepspeed.initialize = lambda *args, **kwargs: (None, None, None, None)
    
    # Register module
    sys.modules["deepspeed"] = deepspeed


if __name__ == "__main__":
    setup_deepspeed_mock()
    print("✅ DeepSpeed mock setup successful")
    
    import deepspeed
    print(f"   Module: {deepspeed}")
