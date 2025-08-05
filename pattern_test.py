"""Create diverse trace patterns for analysis."""

import asyncio
from asc.storage import TraceStorage
from asc.core import TracingAgent

async def create_pattern_data():
    """Generate systematic trace patterns."""
    
    # Initialize clean storage
    storage = TraceStorage()
    await storage.initialize()
    
    # Create agent with storage
    agent = TracingAgent("pattern-test-agent", storage=storage)
    
    print("🔄 Creating systematic trace patterns...")
    
    # Pattern 1: Math -> File (should repeat)
    for i in range(3):
        trace = await _create_trace(agent, f"Math sequence {i}")
        await agent._execute_tool_call("math_calculate", {"expression": f"sqrt({16 + i})"}, trace)
        await agent._execute_tool_call("file_write", {"path": f"math_result_{i}.txt", "content": f"Result {i}"}, trace)
        await agent.storage.store_task_trace(trace)  # Store the complete trace
    
    # Pattern 2: Random -> Transform -> File (should repeat)  
    for i in range(2):
        trace = await _create_trace(agent, f"Random sequence {i}")
        await agent._execute_tool_call("random_number", {"min_val": 1, "max_val": 100}, trace)
        await agent._execute_tool_call("string_transform", {"text": f"Number {i}", "operation": "upper"}, trace)
        await agent._execute_tool_call("file_write", {"path": f"random_{i}.txt", "content": f"Random data {i}"}, trace)
        await agent.storage.store_task_trace(trace)  # Store the complete trace
    
    # Pattern 3: HTTP -> File (should repeat)
    for i in range(2):
        trace = await _create_trace(agent, f"HTTP sequence {i}")
        await agent._execute_tool_call("http_get", {"url": "https://httpbin.org/uuid"}, trace)
        await agent._execute_tool_call("file_write", {"path": f"api_data_{i}.txt", "content": f"API response {i}"}, trace)
        await agent.storage.store_task_trace(trace)  # Store the complete trace
    
    print("✅ Pattern generation complete!")
    
    # Analyze what we created
    count = await storage.get_trace_count()
    recent = await storage.get_recent_traces(20)
    
    print(f"📊 Total tool calls: {count}")
    
    # Group by session to see patterns
    sessions = {}
    for trace in recent:
        session = trace['session_id']
        if session not in sessions:
            sessions[session] = []
        sessions[session].append(trace['tool_name'])
    
    print(f"📈 Found {len(sessions)} sessions:")
    for session_id, tools in sessions.items():
        print(f"  {session_id[:8]}: {' -> '.join(tools)}")
    
    # Count patterns
    from collections import Counter
    pairs = []
    for tools in sessions.values():
        for i in range(len(tools) - 1):
            pairs.append((tools[i], tools[i+1]))
    
    counts = Counter(pairs)
    print(f"🎯 Top patterns:")
    for pattern, count in counts.most_common():
        if count > 1:
            print(f"  {pattern[0]} -> {pattern[1]}: {count} times")

async def _create_trace(agent, description):
    """Helper to create a trace object."""
    from asc.core.models import TaskTrace
    return TaskTrace(
        agent_id=agent.agent_id,
        task_description=description
    )

if __name__ == "__main__":
    asyncio.run(create_pattern_data())
