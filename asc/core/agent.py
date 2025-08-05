"""Core agent implementation with tracing capabilities."""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

import ollama

from asc.core.models import ToolCall, TaskTrace
from asc.tools.registry import ToolRegistry
from asc.config import config

if TYPE_CHECKING:
    from asc.storage.traces import TraceStorage


class TracingAgent:
    """An agent that traces all tool calls for pattern mining."""
    
    def __init__(
        self,
        agent_id: str,
        model_name: str = "llama3.2:3b",
        storage = None,
        tools: Optional[ToolRegistry] = None
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.storage = storage  # Will be passed in to avoid circular import
        self.tools = tools or ToolRegistry()
        
        # Track recent tool calls for context
        self.recent_tools: List[str] = []
        self.max_context_tools = 5
    
    async def execute_task(self, task: str, session_id: Optional[UUID] = None) -> TaskTrace:
        """Execute a task and collect traces."""
        if session_id is None:
            session_id = uuid4()
            
        trace = TaskTrace(
            session_id=session_id,
            agent_id=self.agent_id,
            task_description=task
        )
        
        try:
            # Generate a simple plan based on task type
            plan = self._generate_simple_plan(task)
            
            # Execute each step in the plan
            for step in plan:
                await self._execute_tool_call(
                    step["tool"], 
                    step.get("args", {}), 
                    trace
                )
            
            trace.final_success = True
                
        except Exception as e:
            trace.final_success = False
            print(f"Task execution failed: {e}")
            
        finally:
            trace.end_time = datetime.now()
            
            # Calculate totals
            trace.total_cost = sum(call.cost_estimate for call in trace.tool_calls)
            trace.total_latency_ms = sum(call.latency_ms for call in trace.tool_calls)
            
        # Store the complete trace
        await self.storage.store_task_trace(trace)
        
        return trace
    
    def _generate_simple_plan(self, task: str) -> List[Dict[str, Any]]:
        """Generate a realistic tool sequence based on the task description."""
        task_lower = task.lower()
        
        # Define common patterns based on task type
        if any(word in task_lower for word in ['analyze', 'analysis', 'data', 'report']):
            return [
                {"tool": "read_file", "args": {"filename": "data.csv"}},
                {"tool": "transform_data", "args": {"operation": "clean"}},
                {"tool": "calculate_stats", "args": {"metric": "summary"}},
                {"tool": "write_file", "args": {"filename": "report.txt", "content": "Analysis complete"}}
            ]
        elif any(word in task_lower for word in ['research', 'find', 'gather', 'search']):
            return [
                {"tool": "web_search", "args": {"query": "relevant topics"}},
                {"tool": "read_file", "args": {"filename": "sources.txt"}},
                {"tool": "write_file", "args": {"filename": "research.md", "content": "Research findings"}}
            ]
        elif any(word in task_lower for word in ['debug', 'test', 'code', 'develop']):
            return [
                {"tool": "read_file", "args": {"filename": "code.py"}},
                {"tool": "run_tests", "args": {"test_type": "unit"}},
                {"tool": "write_file", "args": {"filename": "fixed_code.py", "content": "Fixed code"}}
            ]
        elif any(word in task_lower for word in ['write', 'create', 'generate', 'content']):
            return [
                {"tool": "string_transform", "args": {"text": "draft content", "operation": "expand"}},
                {"tool": "write_file", "args": {"filename": "content.txt", "content": "Generated content"}}
            ]
        elif any(word in task_lower for word in ['automate', 'script', 'workflow']):
            return [
                {"tool": "read_file", "args": {"filename": "config.yaml"}},
                {"tool": "transform_data", "args": {"operation": "process"}},
                {"tool": "random_number", "args": {"min_val": 1, "max_val": 100}}
            ]
        else:
            # Default pattern for any task
            return [
                {"tool": "string_transform", "args": {"text": task[:50], "operation": "lower"}},
                {"tool": "random_number", "args": {"min_val": 1, "max_val": 10}}
            ]
    
    async def _execute_plan(self, plan_text: str, trace: TaskTrace):
        """Execute a plan and add tool calls to trace."""
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON array in the response
            json_match = re.search(r'\\[.*\\]', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
            else:
                # Fallback: create a simple plan
                plan = [{"tool": "string_transform", "args": {"text": "Hello", "operation": "upper"}}]
            
            for step in plan:
                if isinstance(step, dict) and "tool" in step:
                    await self._execute_tool_call(
                        step["tool"],
                        step.get("args", {}),
                        trace
                    )
                    
        except Exception as e:
            # Fallback to a simple demonstration call
            await self._execute_tool_call("random_number", {"min_val": 1, "max_val": 10}, trace)
    
    async def _execute_tool_call(self, tool_name: str, args: Dict[str, Any], trace: TaskTrace):
        """Execute a single tool call and record trace."""
        start_time = time.time()
        
        # Create tool call record
        tool_call = ToolCall(
            session_id=trace.session_id,
            agent_id=self.agent_id,
            tool_name=tool_name,
            args=args,
            result=None,
            success=False,
            latency_ms=0,
            cost_estimate=self.tools.get_tool_cost(tool_name),  # Use realistic costs
            preceding_tools=self.recent_tools.copy()
        )
        
        # Execute the tool
        result = await self.tools.call(tool_name, **args)
        
        # Calculate metrics
        end_time = time.time()
        tool_call.latency_ms = (end_time - start_time) * 1000
        tool_call.success = result["success"]
        tool_call.result = result["result"]
        tool_call.error_message = result["error"]
        
        # Update context
        self.recent_tools.append(tool_name)
        if len(self.recent_tools) > self.max_context_tools:
            self.recent_tools.pop(0)
        
        # Add to trace
        trace.tool_calls.append(tool_call)
        
        # Store immediately for streaming analysis
        await self.storage.store_tool_call(tool_call)
        
        return tool_call
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "[]"  # Return empty plan on failure
