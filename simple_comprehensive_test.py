"""Simpler comprehensive test that works with current agent structure."""

import asyncio
import random
import time
from pathlib import Path
from uuid import uuid4

from asc.core.agent import TracingAgent
from asc.storage.traces import TraceStorage
from asc.tools.registry import ToolRegistry
from asc.mining import PatternMiner
from asc.config import config


async def run_comprehensive_test():
    """Run a comprehensive test with the current system."""
    print("\n🚀 Starting Comprehensive Real-World Test")
    print("=" * 60)
    
    # Initialize components
    storage = TraceStorage()
    await storage.initialize()
    await storage.clear_all_traces()
    
    registry = ToolRegistry()
    
    print("🔧 Current Configuration:")
    print(f"   Mining support threshold: {config.mining.min_support_percent}%")
    print(f"   Success rate threshold: {config.mining.min_success_rate:.0%}")  
    print(f"   Cost efficiency threshold: {config.mining.max_cost_ratio:.0%}")
    print(f"   Dynamic cost measurement: {config.system.dynamic_cost_measurement}")
    
    # Phase 1: Generate realistic usage patterns
    print(f"\n📊 Phase 1: Generating Realistic Agent Usage")
    
    scenarios = {
        'data_analysis': [
            "Analyze sales data and create summary",
            "Find correlations in marketing data",
            "Generate forecast from historical trends",
            "Identify top performing products"
        ],
        'research': [
            "Research competitors in AI market",
            "Find academic papers on ML training",
            "Gather info on autonomous vehicles",
            "Create literature review on energy tech"
        ],
        'development': [
            "Debug authentication issues",
            "Optimize database queries",
            "Write unit tests for payment module",
            "Refactor legacy reporting code"
        ]
    }
    
    # Create multiple agents for different scenarios
    agents = {}
    for scenario in scenarios.keys():
        agents[scenario] = TracingAgent(
            agent_id=f"agent_{scenario}",
            storage=storage,
            tools=registry
        )
    
    # Generate diverse traces
    total_traces = 0
    target_traces = 150  # Smaller but still meaningful
    
    for i in range(target_traces):
        # Choose scenario and task
        scenario = random.choice(list(scenarios.keys()))
        task = random.choice(scenarios[scenario])
        
        # Add some variation
        if random.random() < 0.3:
            task = f"Update and improve: {task}"
        elif random.random() < 0.2:
            task = f"Review and validate: {task}"
        
        # Use different success rates to create realistic patterns
        scenario_success_rates = {
            'data_analysis': 0.95,  # Data tasks are usually reliable
            'research': 0.85,       # Research can be hit-or-miss
            'development': 0.80     # Development often has failures
        }
        
        success_rate = scenario_success_rates[scenario] + random.uniform(-0.1, 0.1)
        success_rate = max(0.6, min(0.99, success_rate))  # Keep in reasonable range
        
        try:
            agent = agents[scenario]
            
            # Create a session that might have multiple related tasks
            session_id = uuid4()
            
            # Sometimes do multiple related tasks in same session
            tasks_in_session = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            
            for task_num in range(tasks_in_session):
                task_variant = task
                if task_num > 0:
                    task_variant = f"Continue with {task.lower()}"
                
                await agent.execute_task(task_variant, session_id=session_id)
                total_traces += 1
                
                # Small delay between tasks in session
                await asyncio.sleep(0.01)
        
        except Exception as e:
            print(f"⚠️  Task failed: {e}")
        
        # Progress update
        if (i + 1) % 25 == 0:
            print(f"   Generated {i + 1}/{target_traces} task sessions...")
    
    print(f"✅ Generated {total_traces} realistic traces")
    
    # Phase 2: Analyze patterns
    print(f"\n🔍 Phase 2: Analyzing Trace Patterns")
    
    trace_count = await storage.get_trace_count()
    print(f"   📈 Total traces in database: {trace_count}")
    
    # Phase 3: Mine patterns
    print(f"\n⛏️  Phase 3: Mining Patterns")
    
    miner = PatternMiner(storage)
    
    # Use the simple pattern mining method that returns (pattern, support) tuples
    patterns = await miner.mine_patterns(limit=trace_count)
    
    if patterns:
        print(f"   🎯 Found {len(patterns)} frequent patterns")
        
        # Filter to skill candidates  
        skills = miner.filter_skill_candidates(patterns)
        print(f"   ⭐ Filtered to {len(skills)} skill candidates")
        
        if skills:
            # Show metrics
            avg_support = sum(s.support_count for s in skills) / len(skills)
            avg_success = sum(s.success_rate for s in skills) / len(skills)
            avg_cost = sum(s.avg_cost for s in skills) / len(skills)
            
            print(f"   📊 Average support: {avg_support:.1f} traces")
            print(f"   ✅ Average success rate: {avg_success:.1%}")
            print(f"   💰 Average cost: {avg_cost:.2f}")
            
            # Show top skills
            print("\n🎯 Top Skill Candidates:")
            for i, skill in enumerate(skills[:5]):
                print(f"   {i+1}. {' → '.join(skill.tool_sequence)}")
                print(f"      Support: {skill.support_count}, Success: {skill.success_rate:.1%}, Cost: {skill.avg_cost:.2f}")
            
            # Export results
            export_path = Path("data/comprehensive_test_skills.csv")
            await miner.export_skills_csv(skills, export_path)
            print(f"\n💾 Exported {len(skills)} skills to {export_path}")
        else:
            print("   📉 No skill candidates met quality thresholds")
    else:
        print("   📉 No patterns found")
    
    # Phase 4: Recommendations
    print(f"\n💡 Configuration Recommendations:")
    
    if not patterns:
        print("   📉 No patterns found. Consider:")
        print(f"      - Lowering support threshold (currently {config.mining.min_support_percent}%)")
        print(f"      - Generating more traces ({trace_count} may be insufficient)")
    elif not skills:
        print("   📊 Patterns found but none passed filters. Consider:")
        print(f"      - Lowering success threshold (currently {config.mining.min_success_rate:.0%})")
        print(f"      - Relaxing cost threshold (currently {config.mining.max_cost_ratio:.0%})")
    elif len(skills) < 3:
        print("   📈 Few skills found. Consider:")
        print("      - Generating more diverse data")
        print("      - Slightly relaxing thresholds")
    else:
        print("   ✅ Good skill discovery! System is working well.")
    
    print(f"\n📋 For detailed configuration guidance:")
    print(f"   from asc.config import config")
    print(f"   print(config.get_tuning_guide())")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
