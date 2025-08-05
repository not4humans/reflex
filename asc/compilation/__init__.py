"""
Phase 4: Skill Compilation Module

Converts validated skill candidates into reusable Python macros or LoRA adapters.
"""

from .compiler import SkillCompiler
from .strategies import PythonMacroStrategy, LoRAStrategy

__all__ = [
    "SkillCompiler",
    "PythonMacroStrategy", 
    "LoRAStrategy"
]
