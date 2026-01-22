"""
Transformers Mock Module

Provides minimal mock implementation of HuggingFace transformers
to allow pedagogicalrl to import without the 500MB+ library.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


class MockAutoTokenizer:
    """Mock AutoTokenizer class."""
    
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls()
    
    def encode(self, text, add_special_tokens=True, **kwargs):
        """Estimate token count (roughly 4 chars per token)."""
        if not text:
            return []
        return list(range(len(text) // 4 + 1))
    
    def decode(self, token_ids, **kwargs):
        return "mock decoded text"
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        """Simple chat template - concatenate messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    
    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


class MockAutoModel:
    """Mock AutoModel class."""
    
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls()


class MockAutoConfig:
    """Mock AutoConfig class."""
    
    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        return cls()


def setup_transformers_mock() -> None:
    """
    Inject mock transformers modules into sys.modules.
    """
    if "transformers" in sys.modules and hasattr(sys.modules["transformers"], "_is_mock"):
        return
    
    # Create main transformers module
    transformers = _create_mock_module("transformers")
    transformers._is_mock = True
    
    # Add classes
    transformers.AutoTokenizer = MockAutoTokenizer
    transformers.AutoModel = MockAutoModel
    transformers.AutoModelForCausalLM = MockAutoModel
    transformers.AutoModelForSeq2SeqLM = MockAutoModel
    transformers.AutoConfig = MockAutoConfig
    transformers.PreTrainedModel = MockAutoModel
    transformers.PreTrainedTokenizer = MockAutoTokenizer
    transformers.PreTrainedTokenizerFast = MockAutoTokenizer
    
    # Register
    sys.modules["transformers"] = transformers


if __name__ == "__main__":
    setup_transformers_mock()
    print("[OK] Transformers mock setup successful")
    
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("test")
    print(f"   Tokenizer encode: {t.encode('hello world')}")
