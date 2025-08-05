"""
Skill Compiler - Main compilation orchestrator for Phase 4.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.models import SkillCandidate
from .strategies import CompilationStrategy, PythonMacroStrategy, LoRAStrategy, CompiledSkill


class SkillCompiler:
    """Main compiler that orchestrates skill compilation using different strategies."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the skill compiler."""
        # Load configuration from JSON file
        if config_path is None:
            config_path = "config/default.json"
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Fallback to default config
            self.config = {
                "compilation": {
                    "strategy": "python_macro",
                    "validation_samples": 100,
                    "validation_threshold": 0.98
                },
                "validation": {
                    "unit_test_count": 50,
                    "fuzz_test_count": 25,
                    "unit_test_threshold": 0.95,
                    "fuzz_test_threshold": 0.85,
                    "max_cost_delta": 0.0
                }
            }
        
        self.compilation_config = self.config["compilation"]
        
        # Initialize compilation strategies
        self.strategies: Dict[str, CompilationStrategy] = {
            "python_macro": PythonMacroStrategy(),
            "lora_adapter": LoRAStrategy()
        }
        
        # Set default strategy
        self.default_strategy = self.compilation_config.get("strategy", "python_macro")
        
        # Track compilation results
        self.compiled_skills: List[CompiledSkill] = []
        self.compilation_stats = {
            "total_candidates": 0,
            "successful_compilations": 0,
            "failed_compilations": 0,
            "strategies_used": {},
            "avg_confidence": 0.0,
            "estimated_total_cost_reduction": 0.0
        }
    
    async def compile_candidates(self, candidates: List[SkillCandidate]) -> List[CompiledSkill]:
        """Compile a list of skill candidates into executable skills."""
        if not candidates:
            print("No candidates provided for compilation")
            return []
        
        print(f"🔧 Starting compilation of {len(candidates)} skill candidates...")
        
        compiled_skills = []
        self.compilation_stats["total_candidates"] = len(candidates)
        
        for i, candidate in enumerate(candidates):
            print(f"Compiling candidate {i+1}/{len(candidates)}: {candidate.tool_sequence}")
            
            # Select appropriate strategy
            strategy = self._select_strategy(candidate)
            if not strategy:
                print(f"  ❌ No suitable strategy found")
                self.compilation_stats["failed_compilations"] += 1
                continue
            
            # Attempt compilation
            try:
                compiled_skill = strategy.compile(candidate)
                if compiled_skill:
                    compiled_skills.append(compiled_skill)
                    self.compiled_skills.append(compiled_skill)
                    
                    # Update stats
                    strategy_name = compiled_skill.strategy
                    self.compilation_stats["successful_compilations"] += 1
                    self.compilation_stats["strategies_used"][strategy_name] = \
                        self.compilation_stats["strategies_used"].get(strategy_name, 0) + 1
                    
                    print(f"  ✅ Compiled with {strategy_name} (confidence: {compiled_skill.confidence:.2%})")
                else:
                    print(f"  ❌ Compilation failed")
                    self.compilation_stats["failed_compilations"] += 1
                    
            except Exception as e:
                print(f"  ❌ Compilation error: {e}")
                self.compilation_stats["failed_compilations"] += 1
        
        # Update final stats
        if compiled_skills:
            self.compilation_stats["avg_confidence"] = sum(
                skill.confidence for skill in compiled_skills
            ) / len(compiled_skills)
            
            self.compilation_stats["estimated_total_cost_reduction"] = sum(
                skill.estimated_cost_reduction for skill in compiled_skills
            )
        
        print(f"\n📊 Compilation complete:")
        print(f"  ✅ {self.compilation_stats['successful_compilations']} successful")
        print(f"  ❌ {self.compilation_stats['failed_compilations']} failed")
        print(f"  📈 Average confidence: {self.compilation_stats['avg_confidence']:.2%}")
        print(f"  💰 Estimated total cost reduction: {self.compilation_stats['estimated_total_cost_reduction']:.1%}")
        
        return compiled_skills
    
    def _select_strategy(self, candidate: SkillCandidate) -> Optional[CompilationStrategy]:
        """Select the most appropriate compilation strategy for a candidate."""
        # Try strategies in priority order
        strategy_priority = [
            self.default_strategy,
            "python_macro",
            "lora_adapter"
        ]
        
        # Remove duplicates while preserving order
        strategy_priority = list(dict.fromkeys(strategy_priority))
        
        for strategy_name in strategy_priority:
            strategy = self.strategies.get(strategy_name)
            if strategy and strategy.validate_pattern(candidate.tool_sequence):
                return strategy
        
        return None
    
    async def save_compiled_skills(self, skills: List[CompiledSkill], output_dir: str = "compiled_skills") -> str:
        """Save compiled skills to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual skill files
        skill_files = []
        for skill in skills:
            if skill.strategy == "python_macro":
                filename = f"{skill.name}_{timestamp}.py"
                file_path = output_path / filename
                
                # Add imports and metadata to the Python file
                full_code = self._create_python_skill_file(skill)
                file_path.write_text(full_code)
                skill_files.append(str(file_path))
                
            elif skill.strategy == "lora_adapter":
                filename = f"{skill.name}_{timestamp}.json"
                file_path = output_path / filename
                file_path.write_text(skill.code)
                skill_files.append(str(file_path))
        
        # Save compilation summary
        summary_file = output_path / f"compilation_summary_{timestamp}.json"
        summary_data = {
            "timestamp": timestamp,
            "compilation_stats": self.compilation_stats,
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "strategy": skill.strategy,
                    "confidence": skill.confidence,
                    "estimated_cost_reduction": skill.estimated_cost_reduction,
                    "metadata": skill.metadata
                }
                for skill in skills
            ]
        }
        
        summary_file.write_text(json.dumps(summary_data, indent=2))
        
        print(f"\n💾 Saved {len(skills)} compiled skills to {output_dir}/")
        print(f"   📄 Summary: {summary_file}")
        for file_path in skill_files:
            print(f"   📁 Skill: {file_path}")
        
        return str(output_path)
    
    def _create_python_skill_file(self, skill: CompiledSkill) -> str:
        """Create a complete Python file for a compiled skill."""
        header = f'''"""
Auto-compiled skill: {skill.name}
Generated by Agentic Skill Compiler

Description: {skill.description}
Strategy: {skill.strategy}
Confidence: {skill.confidence:.2%}
Estimated cost reduction: {skill.estimated_cost_reduction:.1%}

Pattern: {" -> ".join(skill.metadata.get("pattern", []))}
Support: {skill.metadata.get("support", 0)}
Success rate: {skill.metadata.get("success_rate", 0):.2%}
Average cost: ${skill.metadata.get("avg_cost", 0):.2f}
"""

import asyncio
from typing import Any, Optional
from ..tools.registry import ToolRegistry


# Initialize tool registry (should be injected in production)
tool_registry = ToolRegistry()


{skill.code}


# Metadata for skill validation and monitoring
SKILL_METADATA = {{
    "name": "{skill.name}",
    "description": "{skill.description}",
    "parameters": {skill.parameters},
    "strategy": "{skill.strategy}",
    "confidence": {skill.confidence},
    "estimated_cost_reduction": {skill.estimated_cost_reduction},
    "pattern": {skill.metadata.get("pattern", [])},
    "support": {skill.metadata.get("support", 0)},
    "success_rate": {skill.metadata.get("success_rate", 0)},
    "avg_cost": {skill.metadata.get("avg_cost", 0)}
}}


if __name__ == "__main__":
    # Example usage
    async def test_skill():
        print(f"Testing compiled skill: {{SKILL_METADATA['name']}}")
        # Add test parameters here
        # result = await {skill.name}(...)
        # print(f"Result: {{result}}")
    
    asyncio.run(test_skill())
'''
        return header
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get current compilation statistics."""
        return self.compilation_stats.copy()
    
    def list_compiled_skills(self) -> List[Dict[str, Any]]:
        """Get a list of all compiled skills with basic info."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "strategy": skill.strategy,
                "confidence": skill.confidence,
                "estimated_cost_reduction": skill.estimated_cost_reduction,
                "parameters": skill.parameters
            }
            for skill in self.compiled_skills
        ]
    
    async def validate_compiled_skill(self, skill: CompiledSkill) -> Dict[str, Any]:
        """Validate a compiled skill (placeholder for Phase 5)."""
        # This will be implemented in Phase 5: Validation
        validation_config = self.config.get("validation", {})
        
        # Basic validation checks
        validation_results = {
            "skill_name": skill.name,
            "syntax_valid": True,  # We already check this during compilation
            "confidence_threshold_met": skill.confidence >= 0.7,
            "estimated_viable": skill.estimated_cost_reduction > 0.1,
            "validation_status": "pending_full_validation",
            "recommendations": []
        }
        
        # Add recommendations based on confidence
        if skill.confidence < 0.8:
            validation_results["recommendations"].append(
                "Consider gathering more training data to improve confidence"
            )
        
        if skill.estimated_cost_reduction < 0.2:
            validation_results["recommendations"].append(
                "Cost reduction may be minimal - verify actual performance"
            )
        
        return validation_results
