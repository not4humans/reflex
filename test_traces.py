"""Simple test tasks that don't require LLM planning for now."""

import asyncio
from asc.core import TracingAgent
from asc.storage import TraceStorage

async def run_manual_tests():
    """Run pre-planned tasks to generate diverse traces."""
    
    storage = TraceStorage()
    await storage.initialize()
    agent = TracingAgent("test-agent", storage=storage)
    
    # Pre-planned task sequences to generate patterns
    test_sequences = [
        # Math operations
        [
            ("math_calculate", {"expression": "sqrt(16)"}),
            ("file_write", {"path": "result1.txt", "content": "Result: 4"}),
        ],
        # Random number generation  
        [
            ("random_number", {"min_val": 50, "max_val": 100}),
            ("string_transform", {"text": "Random: 75.3", "operation": "upper"}),
            ("file_write", {"path": "random.txt", "content": "RANDOM: 75.3"}),
        ],
        # JSON processing
        [
            ("json_parse", {"json_str": '{"name": "test", "value": 42}'}),
            ("string_transform", {"text": "test", "operation": "upper"}),
        ],
        # HTTP and file combo
        [
            ("http_get", {"url": "https://httpbin.org/json"}),
            ("file_write", {"path": "api_response.txt", "content": "API data received"}),
        ],
        # Math chain
        [
            ("math_calculate", {"expression": "10 * 5"}),
            ("math_calculate", {"expression": "sqrt(50)"}),
            ("file_write", {"path": "calc_results.txt", "content": "Math done"}),
        ],
        # Repeat pattern 1 (should be detected)
        [
            ("math_calculate", {"expression": "sqrt(25)"}),
            ("file_write", {"path": "result2.txt", "content": "Result: 5"}),
        ],
        # Repeat pattern 2 (should be detected)
        [
            ("random_number", {"min_val": 1, "max_val": 10}),
            ("string_transform", {"text": "Random: 7", "operation": "upper"}),
            ("file_write", {"path": "random2.txt", "content": "RANDOM: 7"}),
        ],
    ]
    
    print("🧪 Running manual test sequences...")
    
    for i, sequence in enumerate(test_sequences, 1):
        print(f"\n📋 Sequence {i}: {len(sequence)} tools")
        
        # Create a trace for this sequence
        from asc.core.models import TaskTrace
        trace = TaskTrace(
            agent_id=agent.agent_id,
            task_description=f"Manual test sequence {i}"
        )
        
        # Execute each tool in sequence
        for tool_name, args in sequence:
            tool_call = await agent._execute_tool_call(tool_name, args, trace)
            success = "✅" if tool_call.success else "❌"
            print(f"  {success} {tool_name}({list(args.keys())}) -> {str(tool_call.result)[:30]}...")
        
        # Complete the trace
        from datetime import datetime
        trace.end_time = datetime.utcnow()
        trace.final_success = all(call.success for call in trace.tool_calls)
        trace.total_cost = sum(call.cost_estimate for call in trace.tool_calls)
        trace.total_latency_ms = sum(call.latency_ms for call in trace.tool_calls)
        
        # Store trace (this will also store individual tool calls)
        await storage.store_task_trace(trace)
    
    # Show summary
    total_count = await storage.get_trace_count()
    recent_traces = await storage.get_recent_traces(limit=20)
    
    print(f"\n📊 Test complete! Total tool calls: {total_count}")
    print(f"📈 Recent traces collected: {len(recent_traces)}")
    
    # Analyze patterns
    print("\n🔍 Pattern Analysis:")
    tool_sequences = {}
    for trace in recent_traces:
        session = trace['session_id']
        if session not in tool_sequences:
            tool_sequences[session] = []
        tool_sequences[session].append(trace['tool_name'])
    
    print(f"Sessions: {len(tool_sequences)}")
    for session_id, tools in tool_sequences.items():
        print(f"  {session_id[:8]}: {' -> '.join(tools)}")
    
    # Look for repeated 2-tool patterns
    from collections import Counter
    two_tool_patterns = []
    for tools in tool_sequences.values():
        for i in range(len(tools) - 1):
            two_tool_patterns.append((tools[i], tools[i+1]))
    
    pattern_counts = Counter(two_tool_patterns)
    print(f"\n🎯 Most common 2-tool patterns:")
    for pattern, count in pattern_counts.most_common(5):
        if count > 1:
            print(f"  {pattern[0]} -> {pattern[1]}: {count} times")

if __name__ == "__main__":
    asyncio.run(run_manual_tests())
