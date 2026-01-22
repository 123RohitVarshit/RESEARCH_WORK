"""
Configuration Package

This package contains all configuration dataclasses and utilities
for the EPT framework. Uses Hydra for configuration management.
"""

from config.dataclasses import (
    TeacherModelConfig,
    StudentModelConfig,
    JudgeModelConfig,
    RewardModelConfig,
    GenerationConfig,
    EvalConfig,
)

__all__ = [
    "TeacherModelConfig",
    "StudentModelConfig", 
    "JudgeModelConfig",
    "RewardModelConfig",
    "GenerationConfig",
    "EvalConfig",
]
