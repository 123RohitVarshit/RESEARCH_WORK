"""
NVIDIA pynvml Mock

Provides mock implementation of pynvml (NVIDIA GPU monitoring library)
for environments without NVIDIA GPUs.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def setup_pynvml_mock() -> None:
    """Set up pynvml mock in sys.modules."""
    if "pynvml" in sys.modules:
        return
    
    pynvml = types.ModuleType("pynvml")
    pynvml.__spec__ = ModuleSpec(name="pynvml", loader=None)
    
    # Mock common functions
    pynvml.nvmlInit = lambda: None
    pynvml.nvmlShutdown = lambda: None
    pynvml.nvmlDeviceGetCount = lambda: 0
    pynvml.nvmlDeviceGetHandleByIndex = lambda idx: None
    pynvml.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(total=0, free=0, used=0)
    pynvml.nvmlDeviceGetName = lambda handle: b"MockGPU"
    pynvml.nvmlDeviceGetUUID = lambda handle: "mock-uuid"
    pynvml.NVMLError = Exception
    
    sys.modules["pynvml"] = pynvml


if __name__ == "__main__":
    setup_pynvml_mock()
    print("[OK] pynvml mock setup successful")
