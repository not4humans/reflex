"""Basic tools for agents to use during trace collection."""

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import httpx


class ToolRegistry:
    """Registry of available tools with consistent interface."""
    
    def __init__(self):
        self.tools = {
            "file_write": self.file_write,
            "file_read": self.file_read,
            "math_calculate": self.math_calculate,
            "http_get": self.http_get,
            "random_number": self.random_number,
            "json_parse": self.json_parse,
            "string_transform": self.string_transform,
        }
    
    async def call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool and return standardized result."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "result": None,
                "error": f"Unknown tool: {tool_name}"
            }
        
        try:
            result = await self.tools[tool_name](**kwargs)
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def get_tool_cost(self, tool_name: str) -> float:
        """Get the cost estimate for a tool (simulating different computational costs)."""
        # Realistic cost estimates based on computational complexity
        costs = {
            "file_write": 0.1,      # Very cheap
            "file_read": 0.1,       # Very cheap
            "math_calculate": 0.5,  # Medium cost
            "http_get": 2.0,        # Expensive (network I/O)
            "random_number": 0.2,   # Cheap
            "json_parse": 0.3,      # Cheap
            "string_transform": 0.1, # Very cheap
        }
        return costs.get(tool_name, 1.0)
    
    def list_tools(self) -> list:
        """Get list of available tools."""
        return list(self.tools.keys())
    
    # Tool implementations
    async def file_write(self, path: str, content: str) -> str:
        """Write content to a file."""
        file_path = Path("data/sandbox") / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} characters to {path}"
    
    async def file_read(self, path: str) -> str:
        """Read content from a file."""
        file_path = Path("data/sandbox") / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_text()
    
    async def math_calculate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Very basic calculator - only allow safe operations
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e
        }
        
        # Remove any potentially dangerous characters/functions
        if any(char in expression for char in ["import", "exec", "eval", "__"]):
            raise ValueError("Invalid expression")
        
        try:
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")
    
    async def http_get(self, url: str, timeout: float = 5.0) -> Dict[str, Any]:
        """Make an HTTP GET request."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:1000]  # Limit content size
            }
    
    async def random_number(self, min_val: float = 0, max_val: float = 100) -> float:
        """Generate a random number in range."""
        return random.uniform(min_val, max_val)
    
    async def json_parse(self, json_str: str) -> Any:
        """Parse JSON string."""
        return json.loads(json_str)
    
    async def string_transform(self, text: str, operation: str) -> str:
        """Transform string (upper, lower, reverse, etc.)."""
        operations = {
            "upper": text.upper,
            "lower": text.lower,
            "reverse": lambda: text[::-1],
            "strip": text.strip,
            "length": lambda: str(len(text))
        }
        
        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")
        
        return operations[operation]()
