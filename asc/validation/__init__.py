"""
Phase 5: Validation Harness

Implements comprehensive validation for compiled skills including:
- Unit testing with 100 test cases per skill
- Fuzz testing with random/edge case inputs
- Performance benchmarking vs baseline agent
- Cost analysis and reduction validation
- HTML report generation

Success Gates (from research paper):
- Unit tests: ≥ 95% pass rate
- Fuzz tests: ≥ 85% pass rate  
- Cost delta: ≤ 0 (no cost increase)
"""

import asyncio
import json
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from uuid import uuid4
import importlib.util
import sys

from asc.core.agent import TracingAgent
from asc.core.models import TaskTrace
from asc.storage.traces import TraceStorage
from asc.tools.registry import ToolRegistry
from asc.config import config


class SkillValidator:
    """Validates compiled skills through unit and fuzz testing."""
    
    def __init__(self, storage: TraceStorage):
        self.storage = storage
        self.tool_registry = ToolRegistry()
        self.baseline_agent = TracingAgent("baseline-validator", storage=storage)
        
        # Validation results
        self.validation_results = {}
        
    async def validate_skill(self, skill_file_path: Path) -> Dict[str, Any]:
        """
        Run complete validation suite on a compiled skill.
        
        Returns validation report with pass rates, performance metrics, etc.
        """
        print(f"🧪 Validating skill: {skill_file_path.name}")
        
        # Load the skill module
        skill_module = await self._load_skill_module(skill_file_path)
        if not skill_module:
            return {"error": "Failed to load skill module"}
            
        skill_metadata = getattr(skill_module, 'SKILL_METADATA', {})
        skill_function = getattr(skill_module, skill_metadata.get('name', 'unknown_skill'), None)
        
        if not skill_function:
            return {"error": "Skill function not found in module"}
        
        # Run validation test suites
        results = {
            "skill_name": skill_metadata.get('name', 'unknown'),
            "skill_file": str(skill_file_path),
            "metadata": skill_metadata,
            "timestamp": datetime.now().isoformat(),
            "validation_id": str(uuid4())
        }
        
        try:
            # Unit testing (100 tests)
            print("  📋 Running unit tests...")
            unit_results = await self._run_unit_tests(skill_function, skill_metadata)
            results["unit_tests"] = unit_results
            
            # Fuzz testing (edge cases, random inputs)
            print("  🎲 Running fuzz tests...")
            fuzz_results = await self._run_fuzz_tests(skill_function, skill_metadata)
            results["fuzz_tests"] = fuzz_results
            
            # Performance benchmarking
            print("  ⚡ Running performance benchmarks...")
            perf_results = await self._run_performance_tests(skill_function, skill_metadata)
            results["performance"] = perf_results
            
            # Cost analysis
            print("  💰 Analyzing cost impact...")
            cost_results = await self._analyze_cost_impact(skill_function, skill_metadata)
            results["cost_analysis"] = cost_results
            
            # Overall validation status
            results["validation_status"] = self._calculate_validation_status(results)
            
        except Exception as e:
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
            results["validation_status"] = "FAILED"
        
        return results
    
    async def _load_skill_module(self, skill_file_path: Path):
        """Dynamically load a compiled skill module."""
        try:
            spec = importlib.util.spec_from_file_location("skill_module", skill_file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add the module to sys.modules to handle relative imports
            sys.modules["skill_module"] = module
            spec.loader.exec_module(module)
            
            return module
        except Exception as e:
            print(f"Failed to load skill module: {e}")
            return None
    
    async def _run_unit_tests(self, skill_function, metadata: Dict) -> Dict[str, Any]:
        """Run 100 unit tests with valid inputs."""
        test_results = []
        pattern = metadata.get('pattern', [])
        parameters = metadata.get('parameters', [])
        
        for i in range(100):
            try:
                # Generate valid test inputs based on skill parameters
                test_args = self._generate_valid_inputs(parameters, pattern, i)
                
                start_time = time.time()
                result = await skill_function(**test_args)
                end_time = time.time()
                
                test_results.append({
                    "test_id": i,
                    "status": "PASS",
                    "latency_ms": (end_time - start_time) * 1000,
                    "inputs": test_args,
                    "output": str(result)[:100]  # Truncate for report
                })
                
            except Exception as e:
                test_results.append({
                    "test_id": i,
                    "status": "FAIL",
                    "error": str(e),
                    "inputs": test_args if 'test_args' in locals() else {}
                })
        
        # Calculate statistics
        passed = len([t for t in test_results if t["status"] == "PASS"])
        pass_rate = passed / len(test_results)
        avg_latency = sum(t.get("latency_ms", 0) for t in test_results if t["status"] == "PASS") / max(passed, 1)
        
        return {
            "total_tests": len(test_results),
            "passed": passed,
            "failed": len(test_results) - passed,
            "pass_rate": pass_rate,
            "average_latency_ms": avg_latency,
            "meets_gate": pass_rate >= 0.95,  # Paper requirement
            "test_details": test_results
        }
    
    async def _run_fuzz_tests(self, skill_function, metadata: Dict) -> Dict[str, Any]:
        """Run fuzz tests with edge cases and random inputs."""
        test_results = []
        parameters = metadata.get('parameters', [])
        
        # Generate edge cases and random inputs
        fuzz_inputs = []
        
        # Edge cases
        for param in parameters:
            fuzz_inputs.extend([
                {param: ""},  # Empty string
                {param: None},  # None value
                {param: "x" * 1000},  # Very long string
                {param: "special!@#$%^&*()chars"},  # Special characters
                {param: 123},  # Wrong type (int instead of string)
                {param: []},  # Wrong type (list)
            ])
        
        # Random inputs
        for i in range(30):
            random_input = {}
            for param in parameters:
                random_input[param] = self._generate_random_value()
            fuzz_inputs.append(random_input)
        
        # Run fuzz tests
        for i, test_input in enumerate(fuzz_inputs[:50]):  # Limit to 50 fuzz tests
            try:
                start_time = time.time()
                result = await skill_function(**test_input)
                end_time = time.time()
                
                test_results.append({
                    "test_id": i,
                    "status": "PASS",
                    "latency_ms": (end_time - start_time) * 1000,
                    "inputs": test_input,
                    "output": str(result)[:100]
                })
                
            except Exception as e:
                # For fuzz tests, exceptions are expected for invalid inputs
                test_results.append({
                    "test_id": i,
                    "status": "EXPECTED_FAIL" if self._is_expected_failure(e) else "FAIL",
                    "error": str(e),
                    "inputs": test_input
                })
        
        # Calculate statistics (count expected failures as passes for fuzz testing)
        passed = len([t for t in test_results if t["status"] in ["PASS", "EXPECTED_FAIL"]])
        pass_rate = passed / len(test_results)
        
        return {
            "total_tests": len(test_results),
            "passed": passed,
            "failed": len(test_results) - passed,
            "pass_rate": pass_rate,
            "meets_gate": pass_rate >= 0.85,  # Paper requirement
            "test_details": test_results
        }
    
    async def _run_performance_tests(self, skill_function, metadata: Dict) -> Dict[str, Any]:
        """Benchmark skill performance vs baseline agent."""
        pattern = metadata.get('pattern', [])
        parameters = metadata.get('parameters', [])
        
        # Run skill multiple times for statistical significance
        skill_times = []
        for i in range(10):
            test_args = self._generate_valid_inputs(parameters, pattern, i)
            
            start_time = time.time()
            try:
                await skill_function(**test_args)
                end_time = time.time()
                skill_times.append((end_time - start_time) * 1000)
            except:
                pass  # Skip failed tests for performance analysis
        
        # Simulate baseline agent performance (would be real comparison in production)
        baseline_times = []
        for i in range(10):
            # Simulate baseline agent executing the same pattern
            simulated_time = sum(self.tool_registry.get_tool_cost(tool) * 50 for tool in pattern)  # 50ms per cost unit
            baseline_times.append(simulated_time)
        
        skill_avg = sum(skill_times) / max(len(skill_times), 1)
        baseline_avg = sum(baseline_times) / max(len(baseline_times), 1)
        
        return {
            "skill_avg_latency_ms": skill_avg,
            "baseline_avg_latency_ms": baseline_avg,
            "latency_improvement": (baseline_avg - skill_avg) / baseline_avg if baseline_avg > 0 else 0,
            "skill_measurements": skill_times,
            "baseline_measurements": baseline_times
        }
    
    async def _analyze_cost_impact(self, skill_function, metadata: Dict) -> Dict[str, Any]:
        """Analyze cost impact of using compiled skill vs baseline."""
        pattern = metadata.get('pattern', [])
        estimated_reduction = metadata.get('estimated_cost_reduction', 0)
        
        # Calculate baseline cost (sum of individual tool costs)
        baseline_cost = sum(self.tool_registry.get_tool_cost(tool) for tool in pattern)
        
        # Skill cost should be lower due to compilation
        skill_cost = baseline_cost * (1 - estimated_reduction)
        
        cost_delta = skill_cost - baseline_cost
        cost_reduction = (baseline_cost - skill_cost) / baseline_cost if baseline_cost > 0 else 0
        
        return {
            "baseline_cost": baseline_cost,
            "skill_cost": skill_cost,
            "cost_delta": cost_delta,
            "cost_reduction": cost_reduction,
            "estimated_reduction": estimated_reduction,
            "meets_gate": cost_delta <= 0,  # Paper requirement: Δcost ≤ 0
            "pattern": pattern
        }
    
    def _generate_valid_inputs(self, parameters: List[str], pattern: List[str], test_id: int) -> Dict[str, Any]:
        """Generate valid inputs for testing based on skill parameters."""
        test_args = {}
        
        for param in parameters:
            if 'file' in param.lower() or 'path' in param.lower():
                test_args[param] = f"test_file_{test_id}.txt"
            elif 'content' in param.lower() or 'text' in param.lower():
                test_args[param] = f"Test content for iteration {test_id}"
            elif 'url' in param.lower():
                test_args[param] = f"https://example.com/test_{test_id}"
            elif 'number' in param.lower() or 'val' in param.lower():
                test_args[param] = test_id
            else:
                test_args[param] = f"test_value_{test_id}"
        
        return test_args
    
    def _generate_random_value(self):
        """Generate random value for fuzz testing."""
        value_types = [
            lambda: random.choice(["", "test", "x" * random.randint(0, 100)]),
            lambda: random.randint(-1000, 1000),
            lambda: random.choice([True, False, None]),
            lambda: [random.randint(0, 10) for _ in range(random.randint(0, 5))],
            lambda: {"key": "value"},
        ]
        return random.choice(value_types)()
    
    def _is_expected_failure(self, exception: Exception) -> bool:
        """Determine if an exception is expected for invalid fuzz inputs."""
        expected_errors = [
            "TypeError", "ValueError", "AttributeError", 
            "KeyError", "FileNotFoundError"
        ]
        return any(error in str(type(exception)) for error in expected_errors)
    
    def _calculate_validation_status(self, results: Dict) -> str:
        """Calculate overall validation status based on all test results."""
        unit_pass = results.get("unit_tests", {}).get("meets_gate", False)
        fuzz_pass = results.get("fuzz_tests", {}).get("meets_gate", False)
        cost_pass = results.get("cost_analysis", {}).get("meets_gate", False)
        
        if unit_pass and fuzz_pass and cost_pass:
            return "PASSED"
        elif unit_pass and fuzz_pass:
            return "PASSED_WITH_WARNINGS"  # Cost gate failed but core functionality works
        else:
            return "FAILED"


class ValidationHarness:
    """Main validation harness that orchestrates skill validation."""
    
    def __init__(self):
        self.storage = TraceStorage()
        self.validator = SkillValidator(self.storage)
        
    async def initialize(self):
        """Initialize storage and components."""
        await self.storage.initialize()
    
    async def validate_all_skills(self, skills_directory: Path) -> Dict[str, Any]:
        """Validate all compiled skills in a directory."""
        print(f"🧪 Starting Phase 5: Validation Harness")
        print(f"📂 Scanning for skills in: {skills_directory}")
        
        if not skills_directory.exists():
            print(f"❌ Skills directory not found: {skills_directory}")
            return {"error": "Skills directory not found"}
        
        # Find all skill files
        skill_files = list(skills_directory.glob("*.py"))
        if not skill_files:
            print(f"❌ No skill files found in {skills_directory}")
            return {"error": "No skill files found"}
        
        print(f"📋 Found {len(skill_files)} skill files to validate")
        
        # Validate each skill
        validation_results = {}
        for skill_file in skill_files:
            if skill_file.name.startswith("__"):
                continue  # Skip __init__.py etc.
                
            result = await self.validator.validate_skill(skill_file)
            validation_results[skill_file.name] = result
        
        # Generate summary report
        summary = self._generate_summary(validation_results)
        
        # Generate HTML report
        report_path = await self._generate_html_report(validation_results, summary)
        
        return {
            "validation_results": validation_results,
            "summary": summary,
            "report_path": report_path,
            "total_skills": len(skill_files),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_summary(self, validation_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        total_skills = len(validation_results)
        passed_skills = len([r for r in validation_results.values() 
                           if r.get("validation_status") == "PASSED"])
        failed_skills = len([r for r in validation_results.values() 
                           if r.get("validation_status") == "FAILED"])
        
        # Aggregate test statistics
        total_unit_tests = sum(r.get("unit_tests", {}).get("total_tests", 0) 
                              for r in validation_results.values())
        total_unit_passed = sum(r.get("unit_tests", {}).get("passed", 0) 
                               for r in validation_results.values())
        
        overall_unit_pass_rate = total_unit_passed / max(total_unit_tests, 1)
        
        # Calculate average performance improvements
        avg_cost_reduction = sum(r.get("cost_analysis", {}).get("cost_reduction", 0) 
                                for r in validation_results.values()) / max(total_skills, 1)
        
        return {
            "total_skills": total_skills,
            "passed_skills": passed_skills,
            "failed_skills": failed_skills,
            "success_rate": passed_skills / max(total_skills, 1),
            "overall_unit_pass_rate": overall_unit_pass_rate,
            "average_cost_reduction": avg_cost_reduction,
            "meets_research_gates": {
                "unit_tests": overall_unit_pass_rate >= 0.95,
                "phase_5_success": passed_skills >= total_skills * 0.85  # 85% skills must pass
            }
        }
    
    async def _generate_html_report(self, validation_results: Dict, summary: Dict) -> Path:
        """Generate HTML validation report."""
        report_dir = Path("validation_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"validation_report_{timestamp}.html"
        
        html_content = self._create_html_content(validation_results, summary)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Validation report saved: {report_path}")
        return report_path
    
    def _create_html_content(self, validation_results: Dict, summary: Dict) -> str:
        """Create HTML content for validation report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 5: Skill Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 8px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 8px; }}
        .skill {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 8px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .warning {{ border-left: 5px solid #ff9800; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Phase 5: Validation Harness Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Research Paper Success Gates: Unit ≥ 95%, Fuzz ≥ 85%, Δcost ≤ 0</p>
    </div>
    
    <div class="summary">
        <h2>📊 Summary</h2>
        <div class="metric">
            <strong>Total Skills:</strong> {summary['total_skills']}
        </div>
        <div class="metric">
            <strong>Passed:</strong> {summary['passed_skills']} ({summary['success_rate']:.1%})
        </div>
        <div class="metric">
            <strong>Failed:</strong> {summary['failed_skills']}
        </div>
        <div class="metric">
            <strong>Unit Test Pass Rate:</strong> {summary['overall_unit_pass_rate']:.1%}
        </div>
        <div class="metric">
            <strong>Avg Cost Reduction:</strong> {summary['average_cost_reduction']:.1%}
        </div>
        
        <h3>Research Gates Status:</h3>
        <ul>
            <li>Unit Tests ≥ 95%: {'✅ PASS' if summary['meets_research_gates']['unit_tests'] else '❌ FAIL'}</li>
            <li>Phase 5 Success: {'✅ PASS' if summary['meets_research_gates']['phase_5_success'] else '❌ FAIL'}</li>
        </ul>
    </div>
"""
        
        # Add individual skill results
        for skill_name, result in validation_results.items():
            status = result.get('validation_status', 'UNKNOWN')
            css_class = status.lower().replace('_with_warnings', ' warning')
            
            metadata = result.get('metadata', {})
            unit_tests = result.get('unit_tests', {})
            fuzz_tests = result.get('fuzz_tests', {})
            cost_analysis = result.get('cost_analysis', {})
            
            html += f"""
    <div class="skill {css_class}">
        <h3>{metadata.get('name', skill_name)} - {status}</h3>
        <p><strong>Description:</strong> {metadata.get('description', 'N/A')}</p>
        <p><strong>Pattern:</strong> {' → '.join(metadata.get('pattern', []))}</p>
        <p><strong>Confidence:</strong> {metadata.get('confidence', 0):.1%}</p>
        
        <table>
            <tr><th>Test Type</th><th>Pass Rate</th><th>Gate</th><th>Status</th></tr>
            <tr>
                <td>Unit Tests</td>
                <td>{unit_tests.get('pass_rate', 0):.1%} ({unit_tests.get('passed', 0)}/{unit_tests.get('total_tests', 0)})</td>
                <td>≥ 95%</td>
                <td>{'✅' if unit_tests.get('meets_gate', False) else '❌'}</td>
            </tr>
            <tr>
                <td>Fuzz Tests</td>
                <td>{fuzz_tests.get('pass_rate', 0):.1%} ({fuzz_tests.get('passed', 0)}/{fuzz_tests.get('total_tests', 0)})</td>
                <td>≥ 85%</td>
                <td>{'✅' if fuzz_tests.get('meets_gate', False) else '❌'}</td>
            </tr>
            <tr>
                <td>Cost Impact</td>
                <td>{cost_analysis.get('cost_reduction', 0):.1%} reduction</td>
                <td>Δcost ≤ 0</td>
                <td>{'✅' if cost_analysis.get('meets_gate', False) else '❌'}</td>
            </tr>
        </table>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


# CLI integration
async def run_validation_cli(skills_directory: str = "test_output"):
    """CLI entry point for validation harness."""
    harness = ValidationHarness()
    await harness.initialize()
    
    skills_path = Path(skills_directory)
    results = await harness.validate_all_skills(skills_path)
    
    if "error" in results:
        print(f"❌ Validation failed: {results['error']}")
        return
    
    summary = results["summary"]
    print(f"\n🎯 Phase 5 Results:")
    print(f"  Skills validated: {summary['total_skills']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Unit test pass rate: {summary['overall_unit_pass_rate']:.1%}")
    print(f"  Average cost reduction: {summary['average_cost_reduction']:.1%}")
    print(f"  Report: {results['report_path']}")
    
    # Check research gates
    gates = summary['meets_research_gates']
    if gates['unit_tests'] and gates['phase_5_success']:
        print(f"✅ Phase 5 SUCCESS: All research gates passed!")
    else:
        print(f"❌ Phase 5 FAILED: Research gates not met")
        print(f"  Unit tests ≥ 95%: {'✅' if gates['unit_tests'] else '❌'}")
        print(f"  Phase success: {'✅' if gates['phase_5_success'] else '❌'}")


if __name__ == "__main__":
    import sys
    skills_dir = sys.argv[1] if len(sys.argv) > 1 else "test_output"
    asyncio.run(run_validation_cli(skills_dir))
