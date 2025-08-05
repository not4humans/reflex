"""
Compilation strategies for converting skill candidates to executable skills.
"""

import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.models import SkillCandidate, ToolCall
from ..context import enhance_skill_with_context


@dataclass
class CompiledSkill:
    """A compiled skill ready for execution."""
    name: str
    description: str
    parameters: List[str]
    code: str
    strategy: str
    metadata: Dict[str, Any]
    confidence: float
    estimated_cost_reduction: float


class CompilationStrategy(ABC):
    """Abstract base class for compilation strategies."""
    
    @abstractmethod
    def compile(self, candidate: SkillCandidate) -> Optional[CompiledSkill]:
        """Compile a skill candidate into an executable skill."""
        pass
    
    @abstractmethod
    def validate_pattern(self, pattern: List[str]) -> bool:
        """Validate if the pattern is suitable for this compilation strategy."""
        pass
    
    # NEW: Context-aware compilation support
    async def compile_with_context(self, candidate: SkillCandidate, storage=None) -> Optional[CompiledSkill]:
        """Compile skill with context analysis if storage is available."""
        skill = self.compile(candidate)
        
        if skill and storage:
            # Enhance with context analysis
            from ..context import enhance_skill_with_context
            enhanced_metadata = await enhance_skill_with_context(
                skill.metadata, candidate.tool_sequence, storage
            )
            skill.metadata.update(enhanced_metadata)
        
        return skill


class PythonMacroStrategy(CompilationStrategy):
    """Compiles skill candidates into Python macro functions."""
    
    def __init__(self):
        self.strategy_name = "python_macro"
    
    def compile(self, candidate: SkillCandidate) -> Optional[CompiledSkill]:
        """Compile a skill candidate into a Python macro function."""
        try:
            # Extract common pattern from tool sequences
            pattern_analysis = self._analyze_pattern(candidate.tool_sequence)
            if not pattern_analysis:
                return None
            
            # Generate Python function code
            function_code = self._generate_function_code(
                candidate, pattern_analysis
            )
            
            # Validate the generated code
            if not self._validate_code(function_code):
                return None
            
            return CompiledSkill(
                name=self._generate_skill_name(candidate.tool_sequence),
                description=self._generate_description(candidate, pattern_analysis),
                parameters=pattern_analysis["parameters"],
                code=function_code,
                strategy=self.strategy_name,
                metadata={
                    "pattern": candidate.tool_sequence,
                    "support": candidate.support_count,
                    "success_rate": candidate.success_rate,
                    "avg_cost": candidate.avg_cost,
                    "tool_chain": pattern_analysis["tool_chain"]
                },
                confidence=self._calculate_confidence(candidate),
                estimated_cost_reduction=self._estimate_cost_reduction(candidate)
            )
        
        except Exception as e:
            print(f"Compilation failed for pattern {candidate.tool_sequence}: {e}")
            return None
    
    def validate_pattern(self, pattern: List[str]) -> bool:
        """Check if pattern is suitable for Python macro compilation."""
        # Must have at least 2 tools
        if len(pattern) < 2:
            return False
        
        # Pattern should not be too complex (max 6 tools for macro)
        if len(pattern) > 6:
            return False
        
        # Should contain recognizable tool patterns
        tool_types = set()
        for tool in pattern:
            tool_type = self._extract_tool_type(tool)
            if tool_type:
                tool_types.add(tool_type)
        
        # Must have at least 2 different tool types
        return len(tool_types) >= 2
    
    def _analyze_pattern(self, pattern: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze pattern to extract compilation metadata."""
        if not self.validate_pattern(pattern):
            return None
        
        tool_chain = []
        parameters = set()
        
        for tool in pattern:
            tool_info = self._parse_tool(tool)
            if tool_info:
                tool_chain.append(tool_info)
                parameters.update(tool_info.get("parameters", []))
        
        return {
            "tool_chain": tool_chain,
            "parameters": list(parameters),
            "complexity": len(tool_chain),
            "automation_level": self._calculate_automation_level(tool_chain)
        }
    
    def _parse_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Parse tool name to extract metadata."""
        # Basic tool parsing - in practice, this would be more sophisticated
        tool_type = self._extract_tool_type(tool_name)
        if not tool_type:
            return None
        
        # Extract parameters based on tool type
        parameters = []
        if "file" in tool_name.lower():
            parameters.extend(["file_path", "content"])
        if "http" in tool_name.lower() or "api" in tool_name.lower():
            parameters.extend(["url", "data"])
        if "calc" in tool_name.lower() or "math" in tool_name.lower():
            parameters.extend(["expression", "precision"])
        
        return {
            "name": tool_name,
            "type": tool_type,
            "parameters": parameters,
            "is_io": tool_type in ["file", "http", "api", "database"]
        }
    
    def _extract_tool_type(self, tool_name: str) -> Optional[str]:
        """Extract tool type from tool name."""
        tool_lower = tool_name.lower()
        
        if any(word in tool_lower for word in ["file", "read", "write"]):
            return "file"
        elif any(word in tool_lower for word in ["http", "api", "request"]):
            return "http"
        elif any(word in tool_lower for word in ["calc", "math", "compute"]):
            return "computation"
        elif any(word in tool_lower for word in ["json", "parse", "format"]):
            return "data_processing"
        elif any(word in tool_lower for word in ["db", "database", "sql"]):
            return "database"
        
        return "generic"
    
    def _generate_function_code(self, candidate: SkillCandidate, analysis: Dict[str, Any]) -> str:
        """Generate Python function code for the skill."""
        func_name = self._generate_skill_name(candidate.tool_sequence)
        parameters = analysis["parameters"]
        tool_chain = analysis["tool_chain"]
        
        # Build function signature
        param_str = ", ".join(parameters) if parameters else ""
        
        # Build function body
        body_lines = []
        body_lines.append(f'    """')
        body_lines.append(f'    Auto-compiled skill: {self._generate_description(candidate, analysis)}')
        body_lines.append(f'    Pattern: {" -> ".join(candidate.tool_sequence)}')
        body_lines.append(f'    Success rate: {candidate.success_rate:.2%}')
        body_lines.append(f'    Average cost: ${candidate.avg_cost:.2f}')
        body_lines.append(f'    """')
        body_lines.append(f'    ')
        
        # Add tool execution logic
        for i, tool_info in enumerate(tool_chain):
            var_name = f"result_{i}"
            tool_name = tool_info["name"]
            tool_params = self._map_parameters_to_tool(tool_info, parameters)
            
            body_lines.append(f'    # Step {i + 1}: {tool_name}')
            if tool_params:
                param_list = ", ".join([f"{k}={v}" for k, v in tool_params.items()])
                body_lines.append(f'    {var_name} = await tool_registry.execute("{tool_name}", {param_list})')
            else:
                body_lines.append(f'    {var_name} = await tool_registry.execute("{tool_name}")')
            body_lines.append(f'    ')
        
        # Add return statement
        if tool_chain:
            last_result = f"result_{len(tool_chain) - 1}"
            body_lines.append(f'    return {last_result}')
        else:
            body_lines.append(f'    return None')
        
        # Combine into complete function
        code_lines = [
            f"async def {func_name}({param_str}):",
        ] + body_lines
        
        return "\n".join(code_lines)
    
    def _map_parameters_to_tool(self, tool_info: Dict[str, Any], available_params: List[str]) -> Dict[str, str]:
        """Map available parameters to tool parameters."""
        mapping = {}
        tool_params = tool_info.get("parameters", [])
        
        for tool_param in tool_params:
            if tool_param in available_params:
                mapping[tool_param] = tool_param
            else:
                # Try to find a similar parameter
                for param in available_params:
                    if tool_param.lower() in param.lower() or param.lower() in tool_param.lower():
                        mapping[tool_param] = param
                        break
        
        return mapping
    
    def _generate_skill_name(self, pattern: List[str]) -> str:
        """Generate a valid Python function name from pattern."""
        # Extract meaningful words from tool names
        words = []
        for tool in pattern:
            # Split on common separators and extract words
            tool_words = re.findall(r'[a-zA-Z]+', tool.lower())
            words.extend(tool_words[:2])  # Take first 2 words from each tool
        
        # Remove common words and duplicates
        meaningful_words = []
        common_words = {"tool", "execute", "run", "call", "get", "set", "do"}
        
        for word in words:
            if word not in common_words and word not in meaningful_words:
                meaningful_words.append(word)
        
        # Create function name
        if not meaningful_words:
            meaningful_words = ["auto", "skill"]
        
        name = "_".join(meaningful_words[:4])  # Max 4 words
        
        # Ensure valid Python identifier
        if not name[0].isalpha():
            name = "skill_" + name
        
        return name
    
    def _generate_description(self, candidate: SkillCandidate, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable description of the skill."""
        tool_chain = analysis["tool_chain"]
        
        if not tool_chain:
            return "Auto-compiled skill from frequent tool pattern"
        
        # Create description based on tool types
        tool_types = [info["type"] for info in tool_chain]
        type_counts = {}
        for t in tool_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        primary_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        
        if len(primary_types) == 1:
            main_type = primary_types[0][0]
            desc = f"Automated {main_type} workflow"
        else:
            main_types = [t[0] for t in primary_types[:2]]
            desc = f"Combined {' and '.join(main_types)} operation"
        
        desc += f" (pattern frequency: {candidate.support_count}, success rate: {candidate.success_rate:.1%})"
        
        return desc
    
    def _calculate_automation_level(self, tool_chain: List[Dict[str, Any]]) -> float:
        """Calculate how automated this skill can be (0.0 to 1.0)."""
        if not tool_chain:
            return 0.0
        
        # Higher automation for more I/O operations and fewer interactive tools
        io_tools = sum(1 for tool in tool_chain if tool.get("is_io", False))
        automation_score = io_tools / len(tool_chain)
        
        # Boost for longer chains (more work automated)
        length_bonus = min(0.2, len(tool_chain) * 0.05)
        
        return min(1.0, automation_score + length_bonus)
    
    def _validate_code(self, code: str) -> bool:
        """Validate that generated code is syntactically correct."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _calculate_confidence(self, candidate: SkillCandidate) -> float:
        """Calculate confidence in the compiled skill."""
        # Base confidence on success rate and support
        base_confidence = candidate.success_rate * 0.7 + (candidate.support_count / 100) * 0.3
        
        # Boost for patterns with good cost efficiency
        if candidate.avg_cost < 2.0:  # Low cost patterns are more reliable
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _estimate_cost_reduction(self, candidate: SkillCandidate) -> float:
        """Estimate cost reduction from using this compiled skill."""
        # Compiled skills should be faster and more efficient
        # Estimate 20-50% cost reduction based on pattern complexity
        pattern_length = len(candidate.tool_sequence)
        if pattern_length <= 2:
            return 0.2  # 20% reduction for simple patterns
        elif pattern_length <= 4:
            return 0.35  # 35% reduction for medium patterns
        else:
            return 0.5  # 50% reduction for complex patterns


class LoRAStrategy(CompilationStrategy):
    """Compiles skill candidates into LoRA (Low-Rank Adaptation) fine-tuning configurations."""
    
    def __init__(self):
        self.strategy_name = "lora_adapter"
    
    def compile(self, candidate: SkillCandidate) -> Optional[CompiledSkill]:
        """Compile a skill candidate into a LoRA configuration."""
        try:
            if not self.validate_pattern(candidate.tool_sequence):
                return None
            
            # Generate LoRA training configuration
            lora_config = self._generate_lora_config(candidate)
            
            return CompiledSkill(
                name=self._generate_adapter_name(candidate.tool_sequence),
                description=self._generate_description(candidate),
                parameters=["input_text", "context"],
                code=json.dumps(lora_config, indent=2),
                strategy=self.strategy_name,
                metadata={
                    "pattern": candidate.tool_sequence,
                    "support": candidate.support_count,
                    "success_rate": candidate.success_rate,
                    "avg_cost": candidate.avg_cost,
                    "lora_rank": lora_config["rank"],
                    "target_modules": lora_config["target_modules"]
                },
                confidence=self._calculate_confidence(candidate),
                estimated_cost_reduction=self._estimate_cost_reduction(candidate)
            )
        
        except Exception as e:
            print(f"LoRA compilation failed for pattern {candidate.tool_sequence}: {e}")
            return None
    
    def validate_pattern(self, pattern: List[str]) -> bool:
        """Check if pattern is suitable for LoRA adaptation."""
        # LoRA is better for complex, reasoning-heavy patterns
        if len(pattern) < 3:
            return False
        
        # Look for patterns that involve complex reasoning or generation
        reasoning_indicators = [
            "plan", "analyze", "generate", "create", "design", 
            "reason", "infer", "deduce", "solve", "optimize"
        ]
        
        pattern_text = " ".join(pattern).lower()
        return any(indicator in pattern_text for indicator in reasoning_indicators)
    
    def _generate_lora_config(self, candidate: SkillCandidate) -> Dict[str, Any]:
        """Generate LoRA fine-tuning configuration."""
        # Determine complexity-based parameters
        pattern_complexity = len(candidate.tool_sequence)
        
        if pattern_complexity <= 3:
            rank = 8
            alpha = 16
        elif pattern_complexity <= 5:
            rank = 16
            alpha = 32
        else:
            rank = 32
            alpha = 64
        
        return {
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "rank": rank,
            "lora_alpha": alpha,
            "lora_dropout": 0.1,
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "task_description": self._generate_task_description(candidate),
            "training_config": {
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "max_grad_norm": 1.0,
                "weight_decay": 0.001
            },
            "data_preprocessing": {
                "max_length": 2048,
                "pattern_template": self._create_pattern_template(candidate.tool_sequence),
                "examples_needed": max(100, candidate.support * 2)
            }
        }
    
    def _generate_adapter_name(self, pattern: List[str]) -> str:
        """Generate a name for the LoRA adapter."""
        # Extract key words from pattern
        key_words = []
        for tool in pattern:
            words = re.findall(r'[a-zA-Z]+', tool.lower())
            key_words.extend(words[:1])  # Take first word from each tool
        
        # Create adapter name
        name_parts = []
        for word in key_words[:3]:  # Max 3 words
            if word not in ["tool", "execute", "run", "call"]:
                name_parts.append(word)
        
        if not name_parts:
            name_parts = ["skill", "adapter"]
        
        return "_".join(name_parts) + "_lora"
    
    def _generate_description(self, candidate: SkillCandidate) -> str:
        """Generate description for the LoRA adapter."""
        return (f"LoRA adapter for pattern: {' -> '.join(candidate.tool_sequence)} "
                f"(support: {candidate.support_count}, success: {candidate.success_rate:.1%})")
    
    def _generate_task_description(self, candidate: SkillCandidate) -> str:
        """Generate a task description for LoRA training."""
        return (f"Fine-tune model to efficiently execute the tool pattern: "
                f"{' -> '.join(candidate.tool_sequence)}. This pattern occurs "
                f"{candidate.support_count} times with {candidate.success_rate:.1%} success rate.")
    
    def _create_pattern_template(self, pattern: List[str]) -> str:
        """Create a template for training data generation."""
        template_parts = []
        for i, tool in enumerate(pattern):
            template_parts.append(f"Step {i+1}: Use {tool} with {{args_{i}}}")
        
        return "To complete this task:\n" + "\n".join(template_parts) + "\nResult: {final_output}"
    
    def _calculate_confidence(self, candidate: SkillCandidate) -> float:
        """Calculate confidence in the LoRA adapter."""
        # LoRA confidence based on pattern complexity and success rate
        complexity_score = min(1.0, len(candidate.tool_sequence) / 6.0)  # Normalize to 0-1
        success_score = candidate.success_rate
        support_score = min(1.0, candidate.support_count / 50.0)  # Normalize to 0-1
        
        return (complexity_score * 0.3 + success_score * 0.5 + support_score * 0.2)
    
    def _estimate_cost_reduction(self, candidate: SkillCandidate) -> float:
        """Estimate cost reduction from using LoRA adapter."""
        # LoRA adapters can provide significant cost reduction for complex patterns
        pattern_length = len(candidate.tool_sequence)
        if pattern_length <= 3:
            return 0.3  # 30% reduction
        elif pattern_length <= 5:
            return 0.5  # 50% reduction
        else:
            return 0.7  # 70% reduction for very complex patterns
