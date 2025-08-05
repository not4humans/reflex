"""Manual test of skill compilation with known patterns."""

import asyncio
from asc.core.models import SkillCandidate
from asc.compilation import SkillCompiler

async def test_compilation():
    """Test compilation with manually created skill candidates."""
    
    # Create test skill candidates based on our debug output
    candidates = [
        SkillCandidate(
            tool_sequence=['string_transform', 'write_file'],
            support_count=10,
            success_rate=1.0,
            avg_cost=1.5,
            avg_latency_ms=150.0,
            validation_status="ready"
        ),
        SkillCandidate(
            tool_sequence=['string_transform', 'random_number'],
            support_count=8,
            success_rate=1.0,
            avg_cost=1.2,
            avg_latency_ms=120.0,
            validation_status="ready"
        ),
        SkillCandidate(
            tool_sequence=['read_file', 'transform_data', 'calculate_stats', 'write_file'],
            support_count=6,
            success_rate=1.0,
            avg_cost=2.5,
            avg_latency_ms=250.0,
            validation_status="ready"
        )
    ]
    
    print("🧪 Testing Skill Compilation with manual candidates")
    print(f"📊 Created {len(candidates)} test candidates")
    
    # Initialize compiler
    compiler = SkillCompiler()
    
    # Compile skills
    compiled_skills = await compiler.compile_candidates(candidates)
    
    print(f"\n✅ Successfully compiled {len(compiled_skills)} skills!")
    
    # Show results
    for skill in compiled_skills:
        print(f"\n🔧 Compiled Skill: {skill.name}")
        print(f"   Strategy: {skill.strategy}")
        print(f"   Confidence: {skill.confidence:.2%}")
        print(f"   Cost Reduction: {skill.estimated_cost_reduction:.1%}")
        print(f"   Description: {skill.description}")
        print(f"   Code Preview:")
        code_lines = skill.code.split('\n')
        for line in code_lines[:10]:  # Show first 10 lines
            print(f"     {line}")
        if len(code_lines) > 10:
            print(f"     ... ({len(code_lines) - 10} more lines)")
    
    # Save compiled skills
    if compiled_skills:
        saved_path = await compiler.save_compiled_skills(compiled_skills, "test_output")
        print(f"\n💾 Skills saved to: {saved_path}")
    
    # Show compilation stats
    stats = compiler.get_compilation_stats()
    print(f"\n📊 Compilation Statistics:")
    print(f"  Total candidates: {stats['total_candidates']}")
    print(f"  Successful: {stats['successful_compilations']}")
    print(f"  Failed: {stats['failed_compilations']}")
    print(f"  Average confidence: {stats['avg_confidence']:.1%}")
    print(f"  Total cost reduction: {stats['estimated_total_cost_reduction']:.1%}")

if __name__ == "__main__":
    asyncio.run(test_compilation())
