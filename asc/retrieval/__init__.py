"""
Phase 6: Skill Retrieval & Salience Gate

Implements the skill registry and retrieval system that allows agents to:
1. Store compiled skills in a searchable registry
2. Find relevant skills for new tasks based on semantic similarity
3. Gate skill usage based on confidence thresholds
4. Fall back to normal tool planning when no suitable skill exists

Success Gates (from research paper):
- Precision@1 ≥ 90% on 500 held-out queries
- Confidence gating at τ ≥ 0.8
- Small encoder for efficient skill ranking
"""

import asyncio
import json
import pickle
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from asc.core.models import TaskTrace, SkillCandidate
from asc.storage.traces import TraceStorage
from asc.config import config


class CompiledSkill:
    """A compiled skill loaded and ready for execution."""
    
    def __init__(self, skill_id: str, metadata: Dict[str, Any], skill_function, file_path: Path):
        self.skill_id = skill_id
        self.name = metadata.get('name', 'unknown_skill')
        self.description = metadata.get('description', '')
        self.parameters = metadata.get('parameters', [])
        self.pattern = metadata.get('pattern', [])
        self.confidence = metadata.get('confidence', 0.0)
        self.estimated_cost_reduction = metadata.get('estimated_cost_reduction', 0.0)
        self.strategy = metadata.get('strategy', 'unknown')
        self.success_rate = metadata.get('success_rate', 0.0)
        self.avg_cost = metadata.get('avg_cost', 0.0)
        
        # Execution details
        self.skill_function = skill_function
        self.file_path = file_path
        self.created_at = datetime.now()
        self.usage_count = 0
        self.last_used = None
        
        # Performance tracking
        self.execution_history = []
        
    async def execute(self, **kwargs) -> Any:
        """Execute the compiled skill with given parameters."""
        start_time = datetime.now()
        
        try:
            # Filter kwargs to only include skill parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.parameters}
            
            # Execute the skill function
            result = await self.skill_function(**filtered_kwargs)
            
            # Track usage
            self.usage_count += 1
            self.last_used = datetime.now()
            
            # Record execution history
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "success": True,
                "execution_time_ms": execution_time,
                "inputs": filtered_kwargs,
                "output": str(result)[:100]  # Truncated
            })
            
            return result
            
        except Exception as e:
            # Record failed execution
            self.execution_history.append({
                "timestamp": start_time.isoformat(),
                "success": False,
                "error": str(e),
                "inputs": kwargs
            })
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this skill."""
        if not self.execution_history:
            return {"executions": 0}
        
        successful_executions = [ex for ex in self.execution_history if ex.get("success", False)]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(self.execution_history),
            "avg_execution_time_ms": np.mean([ex.get("execution_time_ms", 0) for ex in successful_executions]) if successful_executions else 0,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


class SkillRegistry:
    """Registry for storing and retrieving compiled skills."""
    
    def __init__(self, storage: TraceStorage):
        self.storage = storage
        self.skills: Dict[str, CompiledSkill] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.skill_vectors = None
        self.skill_descriptions = []
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "cache_hits": 0
        }
    
    async def initialize(self):
        """Initialize the skill registry."""
        await self.storage.initialize()
        
    async def load_skills_from_directory(self, skills_directory: Path) -> int:
        """Load all compiled skills from a directory."""
        if not skills_directory.exists():
            print(f"Skills directory not found: {skills_directory}")
            return 0
        
        skill_files = list(skills_directory.glob("*.py"))
        loaded_count = 0
        
        for skill_file in skill_files:
            if skill_file.name.startswith("__"):
                continue  # Skip __init__.py etc.
            
            try:
                skill = await self._load_skill_from_file(skill_file)
                if skill:
                    self.skills[skill.skill_id] = skill
                    loaded_count += 1
                    print(f"📦 Loaded skill: {skill.name}")
                    
            except Exception as e:
                print(f"⚠️ Failed to load skill from {skill_file}: {e}")
        
        # Build search index after loading skills
        if loaded_count > 0:
            self._build_search_index()
        
        print(f"✅ Loaded {loaded_count} skills into registry")
        return loaded_count
    
    async def _load_skill_from_file(self, skill_file: Path) -> Optional[CompiledSkill]:
        """Load a single skill from a Python file."""
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location("skill_module", skill_file)
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to handle imports
            module_name = f"skill_module_{uuid4().hex[:8]}"
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Extract metadata
            metadata = getattr(module, 'SKILL_METADATA', {})
            if not metadata:
                print(f"⚠️ No SKILL_METADATA found in {skill_file}")
                return None
            
            # Extract skill function
            skill_name = metadata.get('name', 'unknown_skill')
            skill_function = getattr(module, skill_name, None)
            
            if not skill_function:
                print(f"⚠️ Skill function '{skill_name}' not found in {skill_file}")
                return None
            
            # Create CompiledSkill instance
            skill_id = f"{skill_name}_{uuid4().hex[:8]}"
            skill = CompiledSkill(skill_id, metadata, skill_function, skill_file)
            
            return skill
            
        except Exception as e:
            print(f"Failed to load skill from {skill_file}: {e}")
            return None
    
    def _build_search_index(self):
        """Build TF-IDF search index for skill retrieval."""
        if not self.skills:
            return
        
        # Prepare skill descriptions for vectorization
        self.skill_descriptions = []
        skill_ids = []
        
        for skill_id, skill in self.skills.items():
            # Combine description, pattern, and parameters for better matching
            text_features = [
                skill.description,
                " ".join(skill.pattern),
                " ".join(skill.parameters),
                skill.name.replace("_", " ")
            ]
            description = " ".join(text_features)
            
            self.skill_descriptions.append(description)
            skill_ids.append(skill_id)
        
        # Build TF-IDF vectors
        try:
            self.skill_vectors = self.vectorizer.fit_transform(self.skill_descriptions)
            print(f"🔍 Built search index for {len(self.skills)} skills")
        except Exception as e:
            print(f"⚠️ Failed to build search index: {e}")
            self.skill_vectors = None
    
    async def find_best_skill(self, task_description: str, confidence_threshold: float = 0.8) -> Optional[CompiledSkill]:
        """
        Find the best matching skill for a task description.
        
        Returns the highest-confidence skill above the threshold, or None.
        """
        self.retrieval_stats["total_queries"] += 1
        
        if not self.skills or self.skill_vectors is None:
            return None
        
        try:
            # Vectorize the task description
            task_vector = self.vectorizer.transform([task_description])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(task_vector, self.skill_vectors)[0]
            
            # Find best matching skill
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # Get the corresponding skill
            skill_ids = list(self.skills.keys())
            best_skill = self.skills[skill_ids[best_idx]]
            
            # Combine similarity score with skill confidence
            combined_confidence = (best_similarity * 0.7) + (best_skill.confidence * 0.3)
            
            print(f"🔍 Best match: {best_skill.name} (similarity: {best_similarity:.3f}, "
                  f"skill_confidence: {best_skill.confidence:.3f}, combined: {combined_confidence:.3f})")
            
            # Check confidence threshold
            if combined_confidence >= confidence_threshold:
                self.retrieval_stats["successful_retrievals"] += 1
                return best_skill
            else:
                print(f"💔 No skill meets confidence threshold {confidence_threshold:.1f}")
                return None
                
        except Exception as e:
            print(f"⚠️ Error during skill retrieval: {e}")
            return None
    
    def get_skills_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the skill registry."""
        if not self.skills:
            return {"total_skills": 0}
        
        total_usage = sum(skill.usage_count for skill in self.skills.values())
        avg_confidence = np.mean([skill.confidence for skill in self.skills.values()])
        
        return {
            "total_skills": len(self.skills),
            "total_usage": total_usage,
            "average_confidence": avg_confidence,
            "retrieval_stats": self.retrieval_stats,
            "skills": [
                {
                    "name": skill.name,
                    "confidence": skill.confidence,
                    "usage_count": skill.usage_count,
                    "pattern": skill.pattern
                }
                for skill in self.skills.values()
            ]
        }
    
    async def save_registry_state(self, output_path: Path):
        """Save registry state for persistence."""
        state = {
            "skills_metadata": {
                skill_id: {
                    "name": skill.name,
                    "description": skill.description,
                    "pattern": skill.pattern,
                    "confidence": skill.confidence,
                    "file_path": str(skill.file_path),
                    "usage_count": skill.usage_count,
                    "performance_stats": skill.get_performance_stats()
                }
                for skill_id, skill in self.skills.items()
            },
            "retrieval_stats": self.retrieval_stats,
            "created_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Registry state saved to: {output_path}")


class SkillAwareAgent:
    """An agent that can use compiled skills when available."""
    
    def __init__(self, agent_id: str, skill_registry: SkillRegistry, 
                 storage: TraceStorage, confidence_threshold: float = 0.8):
        self.agent_id = agent_id
        self.skill_registry = skill_registry
        self.storage = storage
        self.confidence_threshold = confidence_threshold
        
        # Import the base TracingAgent for fallback
        from asc.core.agent import TracingAgent
        self.fallback_agent = TracingAgent(agent_id, storage=storage)
        
        # Performance tracking
        self.execution_stats = {
            "total_tasks": 0,
            "skill_executions": 0,
            "fallback_executions": 0,
            "skill_success_rate": 0.0,
            "fallback_success_rate": 0.0
        }
    
    async def execute_task(self, task: str, session_id: Optional[str] = None) -> TaskTrace:
        """
        Execute a task, using compiled skills when available.
        
        This is the critical Phase 6 functionality that makes the system work end-to-end.
        """
        self.execution_stats["total_tasks"] += 1
        start_time = datetime.now()
        
        print(f"🎯 Executing task: {task}")
        
        # Step 1: Check for matching compiled skill
        matching_skill = await self.skill_registry.find_best_skill(task, self.confidence_threshold)
        
        if matching_skill:
            print(f"⚡ Using compiled skill: {matching_skill.name}")
            return await self._execute_with_skill(task, matching_skill, session_id)
        else:
            print(f"🔧 Falling back to tool planning")
            return await self._execute_with_fallback(task, session_id)
    
    async def _execute_with_skill(self, task: str, skill: CompiledSkill, session_id: Optional[str]) -> TaskTrace:
        """Execute task using a compiled skill."""
        self.execution_stats["skill_executions"] += 1
        
        # Create trace for skill execution
        from asc.core.models import TaskTrace, ToolCall
        from uuid import uuid4
        
        trace = TaskTrace(
            session_id=uuid4() if session_id is None else session_id,
            agent_id=self.agent_id,
            task_description=task
        )
        
        start_time = datetime.now()
        
        try:
            # Generate skill parameters from task (simplified for demo)
            # In a real system, this would be more sophisticated
            skill_params = self._extract_skill_parameters(task, skill.parameters)
            
            # Execute the compiled skill
            skill_start = datetime.now()
            result = await skill.execute(**skill_params)
            skill_end = datetime.now()
            
            # Create a synthetic tool call representing the skill execution
            skill_call = ToolCall(
                session_id=trace.session_id,
                agent_id=self.agent_id,
                tool_name=f"compiled_skill:{skill.name}",
                args=skill_params,
                result=result,
                success=True,
                latency_ms=(skill_end - skill_start).total_seconds() * 1000,
                cost_estimate=skill.avg_cost * (1 - skill.estimated_cost_reduction)
            )
            
            trace.tool_calls.append(skill_call)
            trace.final_success = True
            
        except Exception as e:
            print(f"❌ Skill execution failed: {e}")
            # Create failed tool call
            skill_call = ToolCall(
                session_id=trace.session_id,
                agent_id=self.agent_id,
                tool_name=f"compiled_skill:{skill.name}",
                args={},
                result=None,
                success=False,
                error_message=str(e),
                latency_ms=0,
                cost_estimate=0
            )
            trace.tool_calls.append(skill_call)
            trace.final_success = False
        
        finally:
            trace.end_time = datetime.now()
            trace.total_cost = sum(call.cost_estimate for call in trace.tool_calls)
            trace.total_latency_ms = sum(call.latency_ms for call in trace.tool_calls)
            
            # Store the trace
            await self.storage.store_task_trace(trace)
        
        return trace
    
    async def _execute_with_fallback(self, task: str, session_id: Optional[str]) -> TaskTrace:
        """Execute task using normal tool planning (fallback)."""
        self.execution_stats["fallback_executions"] += 1
        return await self.fallback_agent.execute_task(task, session_id)
    
    def _extract_skill_parameters(self, task: str, skill_parameters: List[str]) -> Dict[str, Any]:
        """
        Extract parameters from task description for skill execution.
        
        This is a simplified version - a real system would use NLP/LLM parsing.
        """
        params = {}
        
        for param in skill_parameters:
            if 'file' in param.lower() or 'path' in param.lower():
                params[param] = "output.txt"  # Default file
            elif 'text' in param.lower() or 'content' in param.lower():
                params[param] = task  # Use task as text content
            elif 'url' in param.lower():
                params[param] = "https://example.com"
            else:
                params[param] = "default_value"
        
        return params
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        total_tasks = self.execution_stats["total_tasks"]
        if total_tasks == 0:
            return self.execution_stats
        
        skill_usage_rate = self.execution_stats["skill_executions"] / total_tasks
        fallback_usage_rate = self.execution_stats["fallback_executions"] / total_tasks
        
        return {
            **self.execution_stats,
            "skill_usage_rate": skill_usage_rate,
            "fallback_usage_rate": fallback_usage_rate
        }


# CLI integration for Phase 6
async def run_skill_registry_demo(skills_directory: str = "compiled_skills"):
    """Demo of skill registry and retrieval functionality."""
    print(f"🚀 Phase 6: Skill Registry & Retrieval Demo")
    
    # Initialize components
    storage = TraceStorage()
    await storage.initialize()
    
    registry = SkillRegistry(storage)
    await registry.initialize()
    
    # Load skills
    skills_path = Path(skills_directory)
    loaded_count = await registry.load_skills_from_directory(skills_path)
    
    if loaded_count == 0:
        print(f"❌ No skills found in {skills_directory}")
        print("Run 'asc compile' first to generate skills!")
        return
    
    # Create skill-aware agent
    agent = SkillAwareAgent("demo-skill-agent", registry, storage)
    
    # Test skill retrieval and execution
    test_tasks = [
        "Transform text and write to file",
        "Calculate math expression and generate random number",
        "Process data and format output",
        "Deploy application to cloud",
        "Analyze logs and generate report"
    ]
    
    print(f"\n🧪 Testing skill retrieval and execution...")
    
    for i, task in enumerate(test_tasks):
        print(f"\n--- Test {i+1}: {task} ---")
        
        try:
            trace = await agent.execute_task(task)
            status = "✅" if trace.final_success else "❌"
            print(f"{status} Result: {len(trace.tool_calls)} tool calls, "
                  f"cost: {trace.total_cost:.2f}, latency: {trace.total_latency_ms:.1f}ms")
        except Exception as e:
            print(f"❌ Execution failed: {e}")
    
    # Show performance summary
    print(f"\n📊 Performance Summary:")
    agent_stats = agent.get_performance_summary()
    print(f"  Total tasks: {agent_stats['total_tasks']}")
    print(f"  Skill usage: {agent_stats['skill_executions']} ({agent_stats.get('skill_usage_rate', 0):.1%})")
    print(f"  Fallback usage: {agent_stats['fallback_executions']} ({agent_stats.get('fallback_usage_rate', 0):.1%})")
    
    registry_stats = registry.get_skills_summary()
    print(f"\n📋 Registry Summary:")
    print(f"  Total skills: {registry_stats['total_skills']}")
    print(f"  Total usage: {registry_stats['total_usage']}")
    print(f"  Average confidence: {registry_stats['average_confidence']:.1%}")
    
    # Save registry state
    state_path = Path("data/skill_registry_state.json")
    await registry.save_registry_state(state_path)


if __name__ == "__main__":
    import sys
    skills_dir = sys.argv[1] if len(sys.argv) > 1 else "compiled_skills"
    asyncio.run(run_skill_registry_demo(skills_dir))
