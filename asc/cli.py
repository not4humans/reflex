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
from .validation import ValidationHarness

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


@app.command()
def validate(
    skills_dir: str = typer.Option("compiled_skills", help="Directory containing compiled skills to validate"),
    generate_report: bool = typer.Option(True, help="Generate HTML validation report")
):
    """Run Phase 5: Skill Validation Harness."""
    asyncio.run(_run_validation(skills_dir, generate_report))


@app.command()
def retrieve(
    task: str = typer.Argument(..., help="Task description to find skills for"),
    skills_dir: str = typer.Option("compiled_skills", help="Directory containing compiled skills"),
    confidence_threshold: float = typer.Option(0.8, help="Minimum confidence threshold for skill selection"),
    execute: bool = typer.Option(False, help="Execute the retrieved skill if found")
):
    """Run Phase 6: Skill Retrieval & Execution."""
    asyncio.run(_run_retrieval(task, skills_dir, confidence_threshold, execute))


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


async def _run_validation(skills_dir: str, generate_report: bool):
    """Internal validation runner."""
    console.print(f"🧪 Starting Phase 5: Validation Harness", style="bold blue")
    console.print(f"📂 Validating skills in: {skills_dir}")
    
    # Initialize validation harness
    harness = ValidationHarness()
    await harness.initialize()
    
    # Check if skills directory exists
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        console.print(f"❌ Skills directory not found: {skills_dir}", style="bold red")
        console.print("Run 'asc compile' first to generate skills!")
        return
    
    # Count skill files
    skill_files = list(skills_path.glob("*.py"))
    if not skill_files:
        console.print(f"❌ No skill files found in {skills_dir}", style="bold red")
        console.print("Run 'asc compile' first to generate skills!")
        return
    
    console.print(f"📋 Found {len(skill_files)} skill files to validate")
    
    # Run validation
    results = await harness.validate_all_skills(skills_path)
    
    if "error" in results:
        console.print(f"❌ Validation failed: {results['error']}", style="bold red")
        return
    
    # Show results
    summary = results["summary"]
    console.print(f"\n🎯 Phase 5 Results:", style="bold green")
    
    # Create results table
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Gate", style="yellow")
    table.add_column("Status", style="bold")
    
    table.add_row(
        "Skills Validated",
        str(summary['total_skills']),
        "-",
        "✅"
    )
    
    table.add_row(
        "Success Rate",
        f"{summary['success_rate']:.1%} ({summary['passed_skills']}/{summary['total_skills']})",
        "≥ 85%",
        "✅" if summary['success_rate'] >= 0.85 else "❌"
    )
    
    table.add_row(
        "Unit Test Pass Rate",
        f"{summary['overall_unit_pass_rate']:.1%}",
        "≥ 95%",
        "✅" if summary['meets_research_gates']['unit_tests'] else "❌"
    )
    
    table.add_row(
        "Avg Cost Reduction",
        f"{summary['average_cost_reduction']:.1%}",
        "> 0%",
        "✅" if summary['average_cost_reduction'] > 0 else "❌"
    )
    
    console.print(table)
    
    # Research gates status
    gates = summary['meets_research_gates']
    console.print(f"\n📊 Research Paper Gates:", style="bold")
    console.print(f"  Unit tests ≥ 95%: {'✅ PASS' if gates['unit_tests'] else '❌ FAIL'}")
    console.print(f"  Phase 5 success: {'✅ PASS' if gates['phase_5_success'] else '❌ FAIL'}")
    
    if gates['unit_tests'] and gates['phase_5_success']:
        console.print(f"\n🎉 Phase 5 SUCCESS: All research gates passed!", style="bold green")
    else:
        console.print(f"\n💔 Phase 5 FAILED: Research gates not met", style="bold red")
    
    # Show individual skill results
    if len(results["validation_results"]) <= 5:  # Show details for small numbers
        console.print(f"\n📋 Individual Skills:", style="bold")
        for skill_name, result in results["validation_results"].items():
            status = result.get('validation_status', 'UNKNOWN')
            metadata = result.get('metadata', {})
            
            status_color = "green" if status == "PASSED" else "red" if status == "FAILED" else "yellow"
            console.print(f"  {metadata.get('name', skill_name)}: {status}", style=status_color)
    
    # Report path
    if generate_report:
        console.print(f"\n📊 Detailed report: {results['report_path']}", style="bold blue")


async def _run_retrieval(task: str, skills_dir: str, confidence_threshold: float, execute: bool):
    """Internal retrieval runner for Phase 6."""
    console.print(f"🔍 Starting Phase 6: Skill Retrieval & Execution", style="bold blue")
    console.print(f"🎯 Task: {task}")
    console.print(f"📂 Skills directory: {skills_dir}")
    console.print(f"🎚️  Confidence threshold: {confidence_threshold:.1f}")
    
    # Initialize components
    storage = TraceStorage()
    await storage.initialize()
    
    # Import the retrieval system
    from .retrieval import SkillRegistry, SkillAwareAgent
    
    registry = SkillRegistry(storage)
    await registry.initialize()
    
    # Load skills
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        console.print(f"❌ Skills directory not found: {skills_dir}", style="bold red")
        console.print("Run 'asc compile' first to generate skills!")
        return
    
    loaded_count = await registry.load_skills_from_directory(skills_path)
    
    if loaded_count == 0:
        console.print(f"❌ No skills found in {skills_dir}", style="bold red")
        console.print("Run 'asc compile' first to generate skills!")
        return
    
    console.print(f"📦 Loaded {loaded_count} skills into registry")
    
    # Find best matching skill (with empty context for now)
    matching_skill = await registry.find_best_skill(
        task, 
        execution_context={},  # Empty context for CLI usage
        confidence_threshold=confidence_threshold
    )
    
    if not matching_skill:
        console.print(f"💔 No skill meets confidence threshold {confidence_threshold:.1f}", style="yellow")
        console.print("Falling back to normal tool planning...")
        
        # Show available skills for reference
        registry_summary = registry.get_skills_summary()
        if registry_summary["skills"]:
            console.print(f"\n📋 Available skills:", style="bold")
            table = Table()
            table.add_column("Name", style="cyan")
            table.add_column("Confidence", style="yellow")
            table.add_column("Pattern", style="white")
            
            for skill_info in registry_summary["skills"]:
                pattern_str = " → ".join(skill_info["pattern"])
                table.add_row(
                    skill_info["name"],
                    f"{skill_info['confidence']:.1%}",
                    pattern_str
                )
            
            console.print(table)
        return
    
    # Show matched skill
    console.print(f"\n✅ Found matching skill: {matching_skill.name}", style="bold green")
    console.print(f"   Confidence: {matching_skill.confidence:.1%}")
    console.print(f"   Pattern: {' → '.join(matching_skill.pattern)}")
    console.print(f"   Description: {matching_skill.description}")
    
    if not execute:
        console.print(f"\n💡 Add --execute to run this skill", style="blue")
        return
    
    # Execute the skill
    console.print(f"\n⚡ Executing skill...", style="bold yellow")
    
    try:
        agent = SkillAwareAgent("cli-agent", registry, storage, confidence_threshold)
        trace = await agent.execute_task(task)
        
        # Show execution results
        if trace.final_success:
            console.print(f"✅ Skill executed successfully!", style="bold green")
        else:
            console.print(f"❌ Skill execution failed", style="bold red")
        
        console.print(f"   Total cost: {trace.total_cost:.2f}")
        console.print(f"   Total latency: {trace.total_latency_ms:.1f}ms")
        console.print(f"   Tool calls: {len(trace.tool_calls)}")
        
        # Show tool call details
        if trace.tool_calls:
            console.print(f"\n📋 Execution details:", style="bold")
            for i, call in enumerate(trace.tool_calls, 1):
                status = "✅" if call.success else "❌"
                console.print(f"   {i}. {call.tool_name}: {status}")
        
        # Show agent performance
        agent_stats = agent.get_performance_summary()
        console.print(f"\n📊 Agent Performance:", style="bold")
        console.print(f"   Skill usage rate: {agent_stats.get('skill_usage_rate', 0):.1%}")
        console.print(f"   Total tasks: {agent_stats['total_tasks']}")
        
    except Exception as e:
        console.print(f"❌ Execution error: {e}", style="bold red")


if __name__ == "__main__":
    app()
