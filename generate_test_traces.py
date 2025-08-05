"""Integration test for end-to-end skill compilation."""

import asyncio
from asc.core import TracingAgent
from asc.storage import TraceStorage

async def generate_repetitive_traces():
    """Generate some repetitive traces for pattern mining."""
    print("🔄 Generating repetitive traces for testing...")
    
    storage = TraceStorage()
    await storage.initialize()
    
    agent = TracingAgent(agent_id="test-agent", model_name="llama3.2:3b", storage=storage)
    
    # Generate multiple traces with the same patterns
    repetitive_tasks = [
        # File processing pattern (repeated 5 times)
        "Read a file and transform the data",
        "Read a file and transform the data", 
        "Read a file and transform the data",
        "Read a file and transform the data",
        "Read a file and transform the data",
        
        # String processing pattern (repeated 4 times)
        "Transform text to uppercase",
        "Transform text to uppercase",
        "Transform text to uppercase", 
        "Transform text to uppercase",
        
        # Simple calculation pattern (repeated 3 times)
        "Calculate a random number and format it",
        "Calculate a random number and format it",
        "Calculate a random number and format it"
    ]
    
    for i, task in enumerate(repetitive_tasks):
        print(f"  Executing task {i+1}/{len(repetitive_tasks)}: {task}")
        trace = await agent.execute_task(task)
        print(f"    ✅ {len(trace.tool_calls)} tools, success: {trace.final_success}")
    
    total_traces = await storage.get_trace_count()
    print(f"✅ Total traces in database: {total_traces}")

if __name__ == "__main__":
    asyncio.run(generate_repetitive_traces())
