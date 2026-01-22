"""
Pedagogicalrl Internal Mocks

This module provides mocks for pedagogicalrl's internal modules that
have heavy dependencies or aren't needed for API-based inference.

IMPORTANT: We do NOT mock the top-level 'src' module because it needs
to remain a package so Python can find src.inference_providers (real code).
Instead, we only mock specific submodules that have heavy dependencies.
"""

import sys
import types
from importlib.machinery import ModuleSpec


def _create_mock_module(name: str) -> types.ModuleType:
    """Create a mock module with a valid ModuleSpec."""
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name=name, loader=None)
    return module


class MockParallelvLLMInference:
    """Mock for vLLM-based local inference (not used with OpenRouter)."""
    def __init__(self, *args, **kwargs):
        pass
    
    def run_batch(self, *args, **kwargs):
        return []
    
    def sleep(self):
        pass
    
    def wake(self):
        pass


class MockInferenceTask:
    """Mock inference task enum."""
    REWARD = "reward"
    GENERATION = "generation"


def setup_pedagogicalrl_mocks() -> None:
    """
    Set up targeted mocks for heavy pedagogicalrl submodules.
    
    We use a surgical approach: only mock the specific modules that
    have heavy dependencies (vllm, utils), not the entire 'src' package.
    This allows src.inference_providers to be imported normally.
    """
    # Don't mock if we're in the pedagogicalrl directory context
    # where real modules exist
    pass  # Mocking is now handled via try/except in classroom.py


if __name__ == "__main__":
    setup_pedagogicalrl_mocks()
    print("[OK] Pedagogicalrl internal mocks setup successful")
