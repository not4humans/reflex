"""Debug script to examine trace patterns in our database."""

import asyncio
from collections import Counter
from asc.storage import TraceStorage

async def debug_traces():
    storage = TraceStorage()
    await storage.initialize()
    
    # Get recent traces
    traces = await storage.get_recent_traces(50)
    
    print(f"Found {len(traces)} recent traces")
    
    # Analyze tool sequences
    tool_sequences = []
    for trace in traces:
        if trace.tool_calls:
            sequence = [call.tool_name for call in trace.tool_calls]
            tool_sequences.append(sequence)
            print(f"Sequence: {sequence} (success: {trace.final_success})")
    
    print(f"\nTotal sequences: {len(tool_sequences)}")
    
    # Count sequence patterns
    sequence_counter = Counter()
    for seq in tool_sequences:
        if len(seq) >= 2:
            sequence_counter[tuple(seq)] += 1
    
    print(f"\nMost common sequences:")
    for seq, count in sequence_counter.most_common(10):
        print(f"  {list(seq)}: {count} times")
    
    # Count individual tools
    tool_counter = Counter()
    for seq in tool_sequences:
        for tool in seq:
            tool_counter[tool] += 1
    
    print(f"\nMost common tools:")
    for tool, count in tool_counter.most_common(10):
        print(f"  {tool}: {count} times")

if __name__ == "__main__":
    asyncio.run(debug_traces())
