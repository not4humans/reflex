"""Configuration management for skill compilation parameters.

This centralizes all tunable parameters with clear explanations of when and why to adjust them.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import json


@dataclass
class MiningConfig:
    """Phase 3: Pattern Mining Configuration"""
    
    # Support threshold: What % of sessions must contain a pattern?
    # Lower = more patterns found, but potentially less reliable
    # Higher = fewer but more reliable patterns
    min_support_percent: float = 2.0
    
    # Success rate threshold: What % of pattern executions must succeed?
    # Lower = accept more risky patterns
    # Higher = only very reliable patterns
    min_success_rate: float = 0.85  # Reduced from 95% for real-world use
    
    # Cost efficiency threshold: Pattern cost vs baseline cost ratio
    # Lower = only very efficient patterns
    # Higher = accept less efficient but still useful patterns  
    max_cost_ratio: float = 0.8  # Increased from 60% for real-world use
    
    # Minimum pattern length (tools in sequence)
    min_pattern_length: int = 2
    
    # Maximum pattern length (prevent overly complex skills)
    max_pattern_length: int = 6
    
    # How many recent traces to analyze
    trace_analysis_limit: int = 10000


@dataclass 
class CompilationConfig:
    """Phase 4: Skill Compilation Configuration"""
    
    # Compilation strategy: "python_macro" or "lora_adapter"
    strategy: str = "python_macro"  # Start with simpler option
    
    # Validation sample size for byte-equivalent testing
    validation_samples: int = 100
    
    # Success threshold for validation (98% from paper)
    validation_threshold: float = 0.98


@dataclass
class ValidationConfig:
    """Phase 5: Validation Harness Configuration"""
    
    # Number of unit tests per skill
    unit_test_count: int = 50  # Reduced from 100 for faster iteration
    
    # Number of fuzz tests per skill  
    fuzz_test_count: int = 25  # Reduced from 100 for faster iteration
    
    # Success thresholds
    unit_test_threshold: float = 0.95
    fuzz_test_threshold: float = 0.85
    
    # Cost increase tolerance (should be ≤ 0)
    max_cost_delta: float = 0.0


@dataclass
class RetrievalConfig:
    """Phase 6: Skill Retrieval Configuration"""
    
    # Confidence threshold for skill selection
    confidence_threshold: float = 0.8
    
    # Number of held-out queries for precision testing
    precision_test_queries: int = 500
    
    # Target precision@1 score
    target_precision: float = 0.9


@dataclass
class SystemConfig:
    """Overall system configuration"""
    
    # Enable/disable real-time cost measurement
    dynamic_cost_measurement: bool = True
    
    # Cost measurement window (samples for moving average)
    cost_measurement_window: int = 100
    
    # Enable/disable automatic skill compilation
    auto_compilation: bool = False
    
    # Minimum traces before mining starts
    min_traces_for_mining: int = 50
    
    # How often to run mining (every N new traces)
    mining_frequency: int = 100
    
    # Maximum number of active skills
    max_active_skills: int = 500
    
    # Skill eviction criteria (usage per week)
    min_weekly_usage: int = 5


class ConfigManager:
    """Manages all configuration with explanations and validation."""
    
    def __init__(self, config_path: Path = Path("config/skill_compiler.json")):
        self.config_path = config_path
        self.mining = MiningConfig()
        self.compilation = CompilationConfig() 
        self.validation = ValidationConfig()
        self.retrieval = RetrievalConfig()
        self.system = SystemConfig()
        
        # Load from file if exists
        self.load()
    
    def load(self):
        """Load configuration from JSON file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                
            # Update configs from loaded data
            for section, config_obj in [
                ('mining', self.mining),
                ('compilation', self.compilation), 
                ('validation', self.validation),
                ('retrieval', self.retrieval),
                ('system', self.system)
            ]:
                if section in data:
                    for key, value in data[section].items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
    
    def save(self):
        """Save current configuration to JSON file."""
        self.config_path.parent.mkdir(exist_ok=True)
        
        data = {
            'mining': self.mining.__dict__,
            'compilation': self.compilation.__dict__,
            'validation': self.validation.__dict__, 
            'retrieval': self.retrieval.__dict__,
            'system': self.system.__dict__
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_tuning_guide(self) -> str:
        """Get a guide for when and how to tune parameters."""
        return """
# Skill Compiler Configuration Tuning Guide

## Phase 3: Pattern Mining

### min_support_percent (currently {mining.min_support_percent}%)
- **Too High**: No patterns found, increase if you have lots of noisy data
- **Too Low**: Too many unreliable patterns, decrease if patterns seem random
- **Tune when**: Initial setup, after major tool changes

### min_success_rate (currently {mining.min_success_rate:.0%})
- **Too High**: No patterns pass filter, decrease in noisy/experimental environments  
- **Too Low**: Unreliable skills created, increase in production environments
- **Tune when**: Moving from development to production

### max_cost_ratio (currently {mining.max_cost_ratio:.0%})
- **Too Low**: No patterns are cost-efficient enough, increase if baseline is already optimized
- **Too High**: Accepting inefficient patterns, decrease if you want only big wins
- **Tune when**: After baseline performance is established

## Phase 5: Validation

### unit_test_count (currently {validation.unit_test_count})
- **Too Low**: Missing edge cases, increase if skills fail in production
- **Too High**: Slow validation, decrease if iteration speed matters more
- **Tune when**: Based on skill complexity and reliability needs

## System

### min_traces_for_mining (currently {system.min_traces_for_mining})
- **Too Low**: Mining on insufficient data, increase if early patterns are unreliable
- **Too High**: Delayed skill discovery, decrease if you want faster feedback
- **Tune when**: Based on your typical usage patterns

### dynamic_cost_measurement (currently {system.dynamic_cost_measurement})
- **Enable**: For real-world deployment with varying costs
- **Disable**: For testing with predictable costs
- **Tune when**: Moving between testing and production environments
        """.format(
            mining=self.mining,
            validation=self.validation, 
            system=self.system
        )


# Global config instance
config = ConfigManager()
