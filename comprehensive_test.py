"""Comprehensive real-world testing for skill compilation.

This generates realistic agent usage patterns and validates the entire loop.
"""

import asyncio
import random
import time
from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

from asc.core.agent import TracingAgent
from asc.core.models import TaskTrace
from asc.tools.registry import ToolRegistry
from asc.storage.traces import TraceStorage
from asc.config import config


class RealWorldTestScenarios:
    """Generate realistic agent usage patterns for testing."""
    
    def __init__(self):
        self.scenarios = {
            'data_analysis': [
                "Analyze the sales data from Q4 and create a summary report",
                "Find correlations between marketing spend and revenue growth", 
                "Generate a forecast for next quarter based on historical trends",
                "Identify top performing products and their key characteristics",
                "Create visualizations for the quarterly business review"
            ],
            'research': [
                "Research competitors in the AI assistant market",
                "Find academic papers on large language model training",
                "Gather information about recent developments in autonomous vehicles", 
                "Create a literature review on sustainable energy technologies",
                "Analyze public sentiment about new technology releases"
            ],
            'development': [
                "Debug the authentication issue in the user login system",
                "Optimize database queries for the product catalog",
                "Write unit tests for the new payment processing module",
                "Refactor the legacy code in the reporting system", 
                "Deploy the updated API to the staging environment"
            ],
            'content_creation': [
                "Write a blog post about emerging technology trends",
                "Create social media content for the product launch",
                "Generate documentation for the new API endpoints",
                "Draft a proposal for the client meeting next week",
                "Create training materials for the customer support team"
            ],
            'automation': [
                "Set up automated backups for the production database",
                "Create a workflow to sync data between different systems",
                "Automate the process of generating monthly reports",
                "Build a script to monitor system health and send alerts",
                "Implement automated testing for the continuous integration pipeline"
            ]
        }
    
    def get_realistic_task_sequence(self, scenario: str, num_tasks: int = 50) -> List[str]:
        """Generate a realistic sequence of related tasks."""
        base_tasks = self.scenarios[scenario]
        tasks = []
        
        for i in range(num_tasks):
            # Sometimes use exact tasks, sometimes variations
            if random.random() < 0.7:
                base_task = random.choice(base_tasks)
                # Add variation
                if random.random() < 0.3:
                    variations = [
                        f"Update the {base_task.lower()}",
                        f"Review and improve {base_task.lower()}", 
                        f"Create a follow-up to {base_task.lower()}",
                        f"Expand on {base_task.lower()}"
                    ]
                    tasks.append(random.choice(variations))
                else:
                    tasks.append(base_task)
            else:
                # Completely new but related task
                tasks.append(f"Handle a related {scenario} task #{i}")
        
        return tasks
    
    def simulate_user_patterns(self, hours: int = 168) -> List[Dict[str, Any]]:
        """Simulate realistic user patterns over time (default: 1 week)."""
        sessions = []
        current_time = time.time()
        
        for hour in range(hours):
            # Realistic work hours (9 AM - 6 PM, weekdays more active)
            hour_of_day = hour % 24
            day_of_week = (hour // 24) % 7
            
            # Activity probability based on time
            if day_of_week < 5:  # Weekday
                if 9 <= hour_of_day <= 18:  # Work hours
                    activity_prob = 0.8
                elif 19 <= hour_of_day <= 22:  # Evening
                    activity_prob = 0.3
                else:  # Night/early morning
                    activity_prob = 0.05
            else:  # Weekend
                if 10 <= hour_of_day <= 20:
                    activity_prob = 0.2
                else:
                    activity_prob = 0.02
            
            if random.random() < activity_prob:
                # Choose scenario based on time of day
                if 9 <= hour_of_day <= 12:
                    scenario_weights = {
                        'data_analysis': 0.4,
                        'research': 0.3, 
                        'development': 0.2,
                        'content_creation': 0.1,
                        'automation': 0.1
                    }
                elif 13 <= hour_of_day <= 17:
                    scenario_weights = {
                        'development': 0.4,
                        'data_analysis': 0.2,
                        'content_creation': 0.2,
                        'research': 0.1,
                        'automation': 0.1
                    }
                else:  # Evening/weekend
                    scenario_weights = {
                        'research': 0.4,
                        'content_creation': 0.3,
                        'automation': 0.2,
                        'data_analysis': 0.1,
                        'development': 0.1
                    }
                
                scenario = random.choices(
                    list(scenario_weights.keys()),
                    weights=list(scenario_weights.values())
                )[0]
                
                # Number of tasks in this session
                task_count = random.choices(
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    weights=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.01, 0.01]
                )[0]
                
                sessions.append({
                    'timestamp': current_time + hour * 3600,
                    'scenario': scenario,
                    'task_count': task_count,
                    'user_id': f"user_{random.randint(1, 10)}"  # 10 different users
                })
        
        return sessions


class ComprehensiveTestRunner:
    """Runs comprehensive real-world tests of the skill compilation system."""
    
    def __init__(self):
        self.scenarios = RealWorldTestScenarios()
        self.storage = TraceStorage()
        self.registry = ToolRegistry()
        
    async def setup(self):
        """Initialize testing environment."""
        await self.storage.initialize()
        
        # Clear existing data for clean test
        await self.storage.clear_all_traces()
        print("🧹 Cleared existing traces for clean test")
    
    async def generate_realistic_traces(self, 
                                      num_sessions: int = 200,
                                      min_traces_per_session: int = 5,
                                      max_traces_per_session: int = 15) -> int:
        """Generate realistic traces from multiple users and scenarios."""
        print(f"🎭 Generating {num_sessions} realistic user sessions...")
        
        # Get realistic user patterns  
        user_sessions = self.scenarios.simulate_user_patterns(hours=168)  # 1 week
        
        total_traces = 0
        
        for i, session in enumerate(user_sessions[:num_sessions]):
            scenario = session['scenario']
            user_id = session['user_id']
            
            # Generate task sequence for this session
            num_tasks = random.randint(min_traces_per_session, max_traces_per_session)
            tasks = self.scenarios.get_realistic_task_sequence(scenario, num_tasks)
            
            # Create agent for this user
            agent = TracingAgent(
                agent_id=f"{user_id}_{scenario}",
                storage=self.storage,
                registry=self.registry
            )
            
            session_id = uuid4()
            
            for task in tasks[:num_tasks]:
                try:
                    # Execute task with realistic variations in success
                    success_rate = random.uniform(0.7, 0.95)  # Realistic failure rates
                    
                    trace = await agent.execute_task(
                        task, 
                        session_id=session_id,
                        simulated_success_rate=success_rate
                    )
                    total_traces += 1
                    
                    # Small delay between tasks in session
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print(f"⚠️  Task failed: {e}")
            
            # Progress update
            if (i + 1) % 20 == 0:
                print(f"   Generated {i + 1}/{len(user_sessions[:num_sessions])} sessions...")
        
        print(f"✅ Generated {total_traces} realistic traces")
        return total_traces
    
    async def run_full_pipeline_test(self):
        """Run complete end-to-end test of the skill compilation pipeline."""
        print("\n🚀 Starting Comprehensive Real-World Test")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Generate realistic data
        print("\n📊 Phase 1: Generating Real-World Trace Data")
        trace_count = await self.generate_realistic_traces(
            num_sessions=100,  # Substantial but manageable
            min_traces_per_session=3,
            max_traces_per_session=12
        )
        
        # Phase 2: Analyze trace patterns
        print(f"\n🔍 Phase 2: Analyzing {trace_count} Traces")
        traces = await self.storage.get_recent_traces(limit=trace_count)
        
        # Show variety metrics
        unique_tools = set()
        unique_sequences = set()
        success_rates = []
        costs = []
        
        for trace in traces:
            tool_sequence = [call.tool_name for call in trace.tool_calls]
            if len(tool_sequence) >= 2:
                unique_sequences.add(tuple(tool_sequence))
            
            for call in trace.tool_calls:
                unique_tools.add(call.tool_name)
                success_rates.append(1.0 if call.success else 0.0)
                costs.append(call.cost_estimate)
        
        print(f"   📈 Unique tools used: {len(unique_tools)}")
        print(f"   🔗 Unique sequences: {len(unique_sequences)}")
        print(f"   ✅ Average success rate: {sum(success_rates)/len(success_rates):.1%}")
        print(f"   💰 Average cost per call: {sum(costs)/len(costs):.2f}")
        
        # Phase 3: Run pattern mining with current config
        print(f"\n⛏️  Phase 3: Mining Patterns (config: {config.mining.min_support_percent}% support)")
        
        from asc.mining import PatternMiner
        miner = PatternMiner(self.storage)
        
        patterns = await miner.mine_patterns(
            limit=trace_count,
            min_support_count=max(2, int(trace_count * config.mining.min_support_percent / 100))
        )
        
        skills = miner.filter_skill_candidates(patterns)
        
        print(f"   🎯 Found {len(patterns)} patterns")
        print(f"   ⭐ Filtered to {len(skills)} skill candidates")
        
        # Show skill quality metrics
        if skills:
            avg_support = sum(s.support_count for s in skills) / len(skills)
            avg_success = sum(s.success_rate for s in skills) / len(skills)
            avg_cost_ratio = sum(s.avg_cost for s in skills) / len(skills)
            
            print(f"   📊 Average support: {avg_support:.1f} traces")
            print(f"   ✅ Average success rate: {avg_success:.1%}")
            print(f"   💰 Average cost ratio: {avg_cost_ratio:.2f}")
            
            # Show some example skills
            print("\n🎯 Top Skill Candidates:")
            for i, skill in enumerate(skills[:3]):
                print(f"   {i+1}. {' → '.join(skill.tool_sequence)}")
                print(f"      Support: {skill.support_count}, Success: {skill.success_rate:.1%}, Cost: {skill.avg_cost:.2f}")
        
        # Phase 4: Export and validate
        print(f"\n💾 Phase 4: Exporting Skills")
        export_path = Path("data/comprehensive_test_skills.csv")
        await miner.export_skills_csv(skills, export_path)
        print(f"   📄 Exported {len(skills)} skills to {export_path}")
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n🏁 Test Complete in {elapsed:.1f}s")
        print("=" * 60)
        
        return {
            'trace_count': trace_count,
            'pattern_count': len(patterns),
            'skill_count': len(skills),
            'unique_tools': len(unique_tools),
            'unique_sequences': len(unique_sequences),
            'avg_success_rate': sum(success_rates)/len(success_rates),
            'avg_cost': sum(costs)/len(costs),
            'elapsed_time': elapsed
        }


async def main():
    """Run comprehensive testing."""
    runner = ComprehensiveTestRunner()
    await runner.setup()
    
    # Show current configuration
    print("🔧 Current Configuration:")
    print(f"   Mining support threshold: {config.mining.min_support_percent}%")
    print(f"   Success rate threshold: {config.mining.min_success_rate:.0%}")  
    print(f"   Cost efficiency threshold: {config.mining.max_cost_ratio:.0%}")
    print(f"   Dynamic cost measurement: {config.system.dynamic_cost_measurement}")
    
    results = await runner.run_full_pipeline_test()
    
    # Recommendations based on results
    print("\n💡 Recommendations:")
    
    if results['skill_count'] == 0:
        print("   📉 No skills found. Consider:")
        print(f"      - Lowering support threshold (currently {config.mining.min_support_percent}%)")
        print(f"      - Lowering success threshold (currently {config.mining.min_success_rate:.0%})")
        print(f"      - Generating more traces ({results['trace_count']} may be insufficient)")
    elif results['skill_count'] < 5:
        print("   📊 Few skills found. Consider:")
        print("      - Lowering thresholds slightly")
        print("      - Generating more diverse traces")
    elif results['skill_count'] > 50:
        print("   📈 Many skills found. Consider:")
        print("      - Raising thresholds for quality")
        print("      - Implementing skill ranking/prioritization")
    else:
        print("   ✅ Good skill discovery rate")
    
    print(f"\n📋 Configuration tuning guide available via:")
    print(f"   print(config.get_tuning_guide())")


if __name__ == "__main__":
    asyncio.run(main())
