"""Command line interface for the Agentic Skill Compiler."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .core import TracingAgent
from .storage import TraceStorage
from .mining import PatternMiner
from .compilation import SkillCompiler

app = typer.Typer(help="Agentic Skill Compiler - Human-centric skill compilation for multi-agent systems")
console = Console()


@app.command()
def demo(
    tasks: int = typer.Option(5, help="Number of demo tasks to run"),
    agent_id: str = typer.Option("demo-agent", help="Agent identifier"),
    model: str = typer.Option("llama3.2:3b", help="Ollama model to use")
):
    """Run a demonstration of trace collection."""
    asyncio.run(_run_demo(tasks, agent_id, model))


async def _run_demo(tasks: int, agent_id: str, model: str):
    """Internal demo runner."""
    console.print(f"🚀 Starting demo with {tasks} tasks using {model}", style="bold green")
    
    # Initialize components
    storage = TraceStorage()
    await storage.initialize()
    
    agent = TracingAgent(agent_id=agent_id, model_name=model, storage=storage)
    
    # Demo tasks
    demo_tasks = [
        "Calculate the square root of 16 and write it to a file called result.txt",
        "Generate a random number between 50 and 100, then transform it to a string",
        "Read the current directory and count the files",
        "Create a JSON object with the current time and save it",
        "Get the weather from a public API and parse the response",
        "Calculate the area of a circle with radius 5",
        "Transform the word 'HELLO' to lowercase and reverse it",
        "Generate 3 random numbers and calculate their average",
        "Write a poem to a file and then read it back",
        "Parse a JSON string and extract specific fields"
    ]
    
    # Execute tasks
    for i in range(min(tasks, len(demo_tasks))):
        task = demo_tasks[i]
        console.print(f"\n📋 Task {i+1}: {task}")
        
        trace = await agent.execute_task(task)
        
        # Show results
        success_emoji = "✅" if trace.final_success else "❌"
        console.print(f"{success_emoji} Completed in {trace.total_latency_ms:.1f}ms with {len(trace.tool_calls)} tool calls")
        
        for call in trace.tool_calls:
            status = "✅" if call.success else "❌"
            console.print(f"  {status} {call.tool_name}({list(call.args.keys())}) -> {str(call.result)[:50]}...")
    
    # Show summary
    total_traces = await storage.get_trace_count()
    console.print(f"\n📊 Demo complete! Total traces collected: {total_traces}", style="bold blue")


@app.command()
def traces():
    """Show trace statistics."""
    asyncio.run(_show_traces())


async def _show_traces():
    """Show trace information."""
    storage = TraceStorage()
    await storage.initialize()
    
    total_count = await storage.get_trace_count()
    recent_traces = await storage.get_recent_traces(limit=10)
    
    console.print(f"📈 Total traces: {total_count}")
    
    if recent_traces:
        table = Table(title="Recent Task Traces")
        table.add_column("Time", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("Task", style="yellow", max_width=30)
        table.add_column("Tools Used", style="blue")
        table.add_column("Success", style="bold")
        table.add_column("Cost", style="red")
        
        for trace in recent_traces:
            success = "✅" if trace.final_success else "❌"
            tool_count = len(trace.tool_calls)
            tools_text = f"{tool_count} tools"
            
            table.add_row(
                trace.start_time.strftime("%H:%M:%S"),
                trace.agent_id,
                trace.task_description[:30] + "..." if len(trace.task_description) > 30 else trace.task_description,
                tools_text,
                success,
                f"{trace.total_cost:.2f}"
            )
        
        console.print(table)
    else:
        console.print("No traces found. Run 'asc demo' to generate some!")


@app.command()
def mine(
    limit: int = typer.Option(10000, help="Number of recent traces to analyze"),
    export_csv: bool = typer.Option(True, help="Export results to CSV")
):
    """Run Phase 3: Pattern mining with PrefixSpan."""
    asyncio.run(_run_mining(limit, export_csv))


async def _run_mining(limit: int, export_csv: bool):
    """Internal mining runner."""
    console.print(f"🔍 Starting Phase 3: Pattern Mining", style="bold blue")
    console.print(f"📊 Analyzing last {limit} traces...")
    
    storage = TraceStorage()
    await storage.initialize()
    
    # Check if we have enough data
    total_traces = await storage.get_trace_count()
    if total_traces < 10:
        console.print("❌ Need at least 10 traces for meaningful mining. Run 'asc demo' first!", style="bold red")
        return
    
    console.print(f"📈 Found {total_traces} total traces")
    
    # Run pattern mining
    miner = PatternMiner(storage)
    patterns = await miner.mine_patterns(limit=limit)
    candidates = miner.filter_skill_candidates(patterns)
    
    if candidates:
        # Export to CSV
        export_path = Path("data/mined_skills.csv")
        await miner.export_skills_csv(candidates, export_path)
        console.print(f"💾 Exported to {export_path}")
    
    if not candidates:
        console.print("💔 No patterns found meeting the criteria", style="yellow")
        console.print("Try generating more diverse traces with 'asc demo --tasks 10'")
        return
    
    # Show results
    console.print(f"\n🎯 Phase 3 Results:", style="bold green")
    
    table = Table(title="Skill Candidates")
    table.add_column("Pattern", style="cyan")
    table.add_column("Support", style="green")
    table.add_column("Success Rate", style="yellow")
    table.add_column("Avg Cost", style="blue")
    table.add_column("Status", style="bold")
    
    for candidate in candidates[:10]:  # Show top 10
        pattern_str = " → ".join(candidate.tool_sequence)
        table.add_row(
            pattern_str,
            str(candidate.support_count),
            f"{candidate.success_rate:.1%}",
            f"{candidate.avg_cost:.2f}",
            candidate.validation_status
        )
    
    console.print(table)
    
    if len(candidates) > 10:
        console.print(f"... and {len(candidates) - 10} more candidates")
    
    if export_csv:
        console.print("📄 Results exported to data/compiled_skills.csv")


@app.command()
def compile(
    limit: int = typer.Option(10000, help="Number of recent traces to analyze for compilation"),
    strategy: str = typer.Option("python_macro", help="Compilation strategy (python_macro or lora_adapter)"),
    output_dir: str = typer.Option("compiled_skills", help="Output directory for compiled skills"),
    min_confidence: float = typer.Option(0.7, help="Minimum confidence threshold for compilation")
):
    """Run Phase 4: Skill Compilation."""
    asyncio.run(_run_compilation(limit, strategy, output_dir, min_confidence))


async def _run_compilation(limit: int, strategy: str, output_dir: str, min_confidence: float):
    """Internal compilation runner."""
    console.print(f"🔧 Starting Phase 4: Skill Compilation", style="bold blue")
    console.print(f"📊 Analyzing last {limit} traces for compilation...")
    
    storage = TraceStorage()
    await storage.initialize()
    
    # Check if we have enough data
    total_traces = await storage.get_trace_count()
    if total_traces < 20:
        console.print("❌ Need at least 20 traces for meaningful compilation. Run 'asc demo' first!", style="bold red")
        return
    
    console.print(f"📈 Found {total_traces} total traces")
    
    # Load test configuration for more lenient thresholds
    import json
    from pathlib import Path
    
    test_config_path = Path("config/test.json")
    if test_config_path.exists():
        with open(test_config_path, 'r') as f:
            test_config = json.load(f)
        mining_config = test_config["mining"]
        console.print("🔧 Using test configuration with lower thresholds for demonstration")
    else:
        # Fallback to default with lower thresholds for demo
        mining_config = {
            "min_support_percent": 1.0,
            "min_success_rate": 0.5,
            "max_cost_ratio": 2.0,
            "min_pattern_length": 2,
            "max_pattern_length": 6
        }
        console.print("🔧 Using demo configuration with lower thresholds")
    
    # First mine patterns to get skill candidates
    console.print("🔍 Mining patterns for skill candidates...")
    miner = PatternMiner(storage, mining_config)
    patterns = await miner.mine_patterns(limit=limit)
    candidates = miner.filter_skill_candidates(patterns)
    
    if not candidates:
        console.print("💔 No skill candidates found meeting the criteria", style="yellow")
        console.print("Try generating more diverse traces with 'asc demo --tasks 20'")
        return
    
    console.print(f"🎯 Found {len(candidates)} skill candidates")
    
    # Initialize compiler with specific strategy
    compiler = SkillCompiler()
    if strategy in compiler.strategies:
        compiler.default_strategy = strategy
        console.print(f"🔧 Using {strategy} compilation strategy")
    else:
        console.print(f"⚠️  Unknown strategy '{strategy}', using default")
    
    # Compile skills
    compiled_skills = await compiler.compile_candidates(candidates)
    
    # Filter by confidence threshold
    high_confidence_skills = [
        skill for skill in compiled_skills 
        if skill.confidence >= min_confidence
    ]
    
    if not high_confidence_skills:
        console.print(f"💔 No skills met confidence threshold of {min_confidence:.1%}", style="yellow")
        if compiled_skills:
            best_confidence = max(skill.confidence for skill in compiled_skills)
            console.print(f"   Best confidence achieved: {best_confidence:.1%}")
            console.print(f"   Try lowering --min-confidence to {best_confidence:.1f}")
        return
    
    console.print(f"✅ {len(high_confidence_skills)} skills met confidence threshold")
    
    # Save compiled skills
    saved_path = await compiler.save_compiled_skills(high_confidence_skills, output_dir)
    
    # Show results
    console.print(f"\n🎯 Phase 4 Results:", style="bold green")
    
    table = Table(title="Compiled Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Strategy", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Cost Reduction", style="blue")
    table.add_column("Description", style="white", max_width=40)
    
    for skill in high_confidence_skills:
        table.add_row(
            skill.name,
            skill.strategy,
            f"{skill.confidence:.1%}",
            f"{skill.estimated_cost_reduction:.1%}",
            skill.description[:40] + "..." if len(skill.description) > 40 else skill.description
        )
    
    console.print(table)
    
    # Show compilation stats
    stats = compiler.get_compilation_stats()
    console.print(f"\n📊 Compilation Statistics:", style="bold")
    console.print(f"  Total candidates processed: {stats['total_candidates']}")
    console.print(f"  Successful compilations: {stats['successful_compilations']}")
    console.print(f"  Failed compilations: {stats['failed_compilations']}")
    console.print(f"  Average confidence: {stats['avg_confidence']:.1%}")
    console.print(f"  Estimated total cost reduction: {stats['estimated_total_cost_reduction']:.1%}")
    
    if stats['strategies_used']:
        console.print("  Strategies used:")
        for strategy_name, count in stats['strategies_used'].items():
            console.print(f"    {strategy_name}: {count} skills")
    
    console.print(f"\n💾 Skills saved to: {saved_path}", style="bold green")


if __name__ == "__main__":
    app()
