"""
Demo script showing exactly how unit tests are generated for skills.
"""

import asyncio
from asc.validation import SkillValidator
from asc.storage.traces import TraceStorage


async def demo_unit_test_generation():
    """Show exactly how unit tests are generated."""
    
    storage = TraceStorage()
    await storage.initialize()
    validator = SkillValidator(storage)
    
    # Example skill metadata (like what comes from a compiled skill)
    skill_metadata = {
        "name": "string_transform_file_write",
        "description": "Transform text and write to file", 
        "parameters": ['text', 'file_path', 'operation'],
        "pattern": ['string_transform', 'file_write'],
        "confidence": 0.85,
        "estimated_cost_reduction": 0.25
    }
    
    print("🧪 Unit Test Generation Demo")
    print("=" * 50)
    print(f"Skill: {skill_metadata['name']}")
    print(f"Parameters: {skill_metadata['parameters']}")
    print()
    
    # Generate first 10 test inputs to show the pattern
    print("📋 Generated Test Inputs (first 10 of 100):")
    print("-" * 40)
    
    for i in range(10):
        test_args = validator._generate_valid_inputs(
            skill_metadata['parameters'], 
            skill_metadata['pattern'], 
            i
        )
        print(f"Test {i}: {test_args}")
    
    print()
    print("🎲 Fuzz Test Examples:")
    print("-" * 20)
    
    # Show some fuzz test inputs
    parameters = skill_metadata['parameters']
    fuzz_examples = []
    
    for param in parameters[:3]:  # Show first 3 params
        fuzz_examples.extend([
            {param: ""},  # Empty string
            {param: None},  # None value
            {param: "x" * 20},  # Long string (truncated for display)
            {param: 123},  # Wrong type
        ])
    
    for i, fuzz_input in enumerate(fuzz_examples[:8]):
        print(f"Fuzz {i}: {fuzz_input}")
    
    print()
    print("📊 How Tests Are Executed:")
    print("-" * 25)
    print("1. Load skill module dynamically")
    print("2. Extract skill function and metadata")
    print("3. Generate 100 valid test inputs")
    print("4. Execute: result = await skill_function(**test_args)")
    print("5. Measure latency and capture results/errors")
    print("6. Generate 50 fuzz tests with edge cases")
    print("7. Calculate pass rate and validate against ≥95% gate")
    print("8. Generate detailed HTML report")


if __name__ == "__main__":
    asyncio.run(demo_unit_test_generation())
