"""Demo script showing configuration tuning for skill discovery."""

import asyncio
from asc.config import config
from asc.storage.traces import TraceStorage
from asc.mining import PatternMiner


async def demo_configuration_tuning():
    """Demonstrate how to tune configuration for better skill discovery."""
    
    print("🎛️  Configuration Tuning Demo")
    print("=" * 50)
    
    storage = TraceStorage()
    await storage.initialize()
    
    trace_count = await storage.get_trace_count()
    print(f"📊 Current traces in database: {trace_count}")
    
    if trace_count < 10:
        print("❌ Not enough traces for meaningful analysis")
        return
    
    # Try different support thresholds
    print(f"\n🔍 Testing Different Support Thresholds:")
    
    original_support = config.mining.min_support_percent
    
    for support_percent in [0.5, 1.0, 2.0, 5.0]:
        config.mining.min_support_percent = support_percent
        
        miner = PatternMiner(storage)
        patterns = await miner.mine_patterns(limit=trace_count)
        
        print(f"   {support_percent}% support: {len(patterns)} patterns found")
        
        if patterns:
            # Try filtering with different thresholds
            original_success = config.mining.min_success_rate
            original_cost = config.mining.max_cost_ratio
            
            # Relaxed filtering for demonstration
            config.mining.min_success_rate = 0.7  # 70% instead of 85%
            config.mining.max_cost_ratio = 1.2   # Allow 120% cost (no savings required)
            
            skills = miner.filter_skill_candidates(patterns)
            print(f"     → {len(skills)} skills with relaxed filters")
            
            # Show some examples
            if skills:
                print(f"     Examples:")
                for i, skill in enumerate(skills[:3]):
                    print(f"       {i+1}. {' → '.join(skill.tool_sequence)} (support: {skill.support_count})")
            
            # Restore original thresholds
            config.mining.min_success_rate = original_success
            config.mining.max_cost_ratio = original_cost
            
            break  # Stop after finding patterns
    
    # Restore original support
    config.mining.min_support_percent = original_support
    
    print(f"\n💡 Recommendations:")
    if trace_count < 100:
        print(f"   📈 Generate more traces (currently {trace_count}, recommend 200+)")
    
    print(f"   🎯 For development/testing:")
    print(f"      - Support threshold: 0.5-1.0% (find more patterns)")
    print(f"      - Success rate: 70-80% (accept some failures)")  
    print(f"      - Cost ratio: 1.0-1.2 (don't require savings)")
    
    print(f"   🏭 For production:")
    print(f"      - Support threshold: 2-5% (reliable patterns only)")
    print(f"      - Success rate: 90-95% (high reliability)")
    print(f"      - Cost ratio: 0.6-0.8 (require significant savings)")


if __name__ == "__main__":
    asyncio.run(demo_configuration_tuning())
