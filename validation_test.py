"""Quick validation test after cleanup."""

import asyncio
from asc.core.agent import TracingAgent
from asc.storage.traces import TraceStorage


async def generate_test_patterns():
    """Generate clear patterns for validation."""
    storage = TraceStorage()
    await storage.initialize()
    agent = TracingAgent("validation-test", storage=storage)
    
    print("🔄 Generating test patterns...")
    
    # Pattern: Transform + Write (repeat 8 times)
    for i in range(8):
        await agent.execute_task(f"Transform text and write to file {i}")
    
    # Pattern: Calculate + Random (repeat 6 times)  
    for i in range(6):
        await agent.execute_task(f"Calculate result and generate random {i}")
    
    print("✅ Test patterns generated")


if __name__ == "__main__":
    asyncio.run(generate_test_patterns())
