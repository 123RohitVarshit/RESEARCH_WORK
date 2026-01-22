"""
PyTorch Mock Module

Provides minimal mock implementation of PyTorch to allow pedagogicalrl
to import without having torch installed.

PyTorch is a heavy dependency (~2GB) that's only used for local training.
Since we use API-based inference, we can mock it.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


class MockTensor:
    """Mock Tensor class."""
    def __init__(self, *args, **kwargs):
        self.data = args[0] if args else []
    
    def __repr__(self):
        return f"MockTensor({self.data})"
    
    def to(self, *args, **kwargs):
        return self
    
    def cuda(self, *args, **kwargs):
        return self
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
    def item(self):
        return 0
    
    def size(self, *args):
        return (0,)
    
    @property
    def shape(self):
        return (0,)
    
    @property
    def device(self):
        return "cpu"


class MockModule:
    """Mock nn.Module class."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return MockTensor()
    
    def to(self, *args, **kwargs):
        return self
    
    def eval(self):
        return self
    
    def train(self, mode=True):
        return self
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, *args, **kwargs):
        pass


def setup_torch_mock() -> None:
    """
    Inject mock PyTorch modules into sys.modules.
    """
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "_is_mock"):
        return
    
    # Create main torch module
    torch = _create_mock_module("torch")
    torch._is_mock = True
    
    # Add common attributes
    torch.Tensor = MockTensor
    torch.tensor = lambda *args, **kwargs: MockTensor(*args, **kwargs)
    torch.zeros = lambda *args, **kwargs: MockTensor()
    torch.ones = lambda *args, **kwargs: MockTensor()
    torch.randn = lambda *args, **kwargs: MockTensor()
    torch.rand = lambda *args, **kwargs: MockTensor()
    torch.arange = lambda *args, **kwargs: MockTensor()
    torch.cat = lambda *args, **kwargs: MockTensor()
    torch.stack = lambda *args, **kwargs: MockTensor()
    torch.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
    torch.inference_mode = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.long = "long"
    torch.int = "int"
    torch.bool = "bool"
    torch.device = lambda x: x
    torch.cuda = _create_mock_module("torch.cuda")
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    
    # torch.nn
    torch_nn = _create_mock_module("torch.nn")
    torch_nn.Module = MockModule
    torch_nn.Linear = MockModule
    torch_nn.Embedding = MockModule
    torch_nn.LayerNorm = MockModule
    torch_nn.Dropout = MockModule
    torch_nn.functional = _create_mock_module("torch.nn.functional")
    torch.nn = torch_nn
    
    # torch.optim
    torch_optim = _create_mock_module("torch.optim")
    torch_optim.Adam = MockModule
    torch_optim.AdamW = MockModule
    torch_optim.SGD = MockModule
    torch.optim = torch_optim
    
    # torch.utils
    torch_utils = _create_mock_module("torch.utils")
    torch_utils_data = _create_mock_module("torch.utils.data")
    torch_utils_data.DataLoader = lambda *args, **kwargs: []
    torch_utils_data.Dataset = object
    torch_utils.data = torch_utils_data
    torch.utils = torch_utils
    
    # Register all modules
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch_nn
    sys.modules["torch.nn.functional"] = torch_nn.functional
    sys.modules["torch.optim"] = torch_optim
    sys.modules["torch.cuda"] = torch.cuda
    sys.modules["torch.utils"] = torch_utils
    sys.modules["torch.utils.data"] = torch_utils_data


if __name__ == "__main__":
    setup_torch_mock()
    print("✅ PyTorch mock setup successful")
    
    import torch
    t = torch.tensor([1, 2, 3])
    print(f"   Created tensor: {t}")
