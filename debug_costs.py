"""Debug cost calculations to understand why patterns are filtered."""

import asyncio
from asc.storage import TraceStorage
from asc.mining import PatternMiner

async def debug_costs():
    storage = TraceStorage()
    await storage.initialize()
    
    traces = await storage.get_recent_traces(100)
    
    # Check individual tool costs
    print("💰 Individual tool costs:")
    for trace in traces[:10]:
        print(f"  {trace['tool_name']}: {trace['cost_estimate']}")
    
    # Check baseline calculation
    miner = PatternMiner(storage)
    session_data = miner._group_by_session(traces)
    
    print(f"\\n📊 Session analysis:")
    for session_id, calls in list(session_data.items())[:5]:
        total_cost = sum(call['cost_estimate'] for call in calls)
        tools = [call['tool_name'] for call in calls]
        print(f"  {session_id[:8]}: {' → '.join(tools)} | Cost: {total_cost}")
    
    # Calculate baseline
    total_cost = sum(trace['cost_estimate'] for trace in traces)
    print(f"\\n📈 Total cost: {total_cost}, Total calls: {len(traces)}")
    print(f"📊 Baseline cost per tool: {total_cost / len(traces):.2f}")
    
    # Show what 60% of baseline would be
    baseline = total_cost / len(traces)
    threshold = baseline * 0.6
    print(f"🎯 60% threshold: {threshold:.2f}")

if __name__ == "__main__":
    asyncio.run(debug_costs())
