"""Test skill for post-cleanup validation."""

import asyncio


class MockToolRegistry:
    async def execute(self, tool_name: str, **kwargs):
        if tool_name == "string_transform":
            return kwargs.get("text", "").upper()
        elif tool_name == "file_write":
            return {"status": "success", "bytes": len(kwargs.get("content", ""))}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


tool_registry = MockToolRegistry()


async def test_skill(text: str, file_path: str):
    """Test skill for validation."""
    result1 = await tool_registry.execute("string_transform", text=text)
    result2 = await tool_registry.execute("file_write", file_path=file_path, content=result1)
    return result2


SKILL_METADATA = {
    "name": "test_skill",
    "description": "Test skill for post-cleanup validation",
    "parameters": ["text", "file_path"],
    "strategy": "python_macro",
    "confidence": 0.9,
    "estimated_cost_reduction": 0.3,
    "pattern": ["string_transform", "file_write"],
    "support": 10,
    "success_rate": 1.0,
    "avg_cost": 1.5
}
