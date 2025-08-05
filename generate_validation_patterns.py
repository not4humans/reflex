"""Generate repetitive traces specifically for skill compilation testing."""

import asyncio
from asc.core.agent import TracingAgent
from asc.storage.traces import TraceStorage


async def generate_repetitive_patterns():
    """Generate traces with clear repetitive patterns for validation testing."""
    storage = TraceStorage()
    await storage.initialize()
    agent = TracingAgent("pattern-generator", storage=storage)
    
    print("🔄 Generating repetitive patterns for skill compilation...")
    
    # Pattern 1: string_transform + file_write (10 times)
    for i in range(10):
        trace = await agent.execute_task(f"Write content to file {i}")
        print(f"✅ Pattern 1 iteration {i+1}: {len(trace.tool_calls)} tools")
    
    # Pattern 2: math_calculate + string_transform (8 times)
    for i in range(8):
        trace = await agent.execute_task(f"Calculate and format result {i}")
        print(f"✅ Pattern 2 iteration {i+1}: {len(trace.tool_calls)} tools")
    
    # Pattern 3: random_number + string_transform + file_write (6 times)
    for i in range(6):
        trace = await agent.execute_task(f"Generate random data and save {i}")
        print(f"✅ Pattern 3 iteration {i+1}: {len(trace.tool_calls)} tools")
    
    total_traces = await storage.get_trace_count()
    print(f"\n📊 Total traces after generation: {total_traces}")
    print("Ready for pattern mining and compilation!")


if __name__ == "__main__":
    asyncio.run(generate_repetitive_patterns())
