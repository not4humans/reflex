"""Pattern mining and skill candidate identification."""

import asyncio
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from prefixspan import PrefixSpan

from asc.core.models import SkillCandidate, TaskTrace, ToolCall
from asc.storage.traces import TraceStorage
from asc.config import config


class PatternMiner:
    """Mines patterns from execution traces to identify skill candidates."""
    
    def __init__(self, storage: TraceStorage):
        self.storage = storage
    
    async def mine_patterns(self, limit: int = 1000, min_support_count: int = None) -> List[Tuple[List[str], int]]:
        """Mine frequent tool sequences from traces."""
        
        # Use config for min support if not provided
        if min_support_count is None:
            min_support_count = max(2, int(limit * config.mining.min_support_percent / 100))
        
        print(f"🔍 Mining patterns with support >= {min_support_count}")
        
        # Get recent traces
        traces = await self.storage.get_recent_traces(limit=limit)
        
        # Extract tool sequences
        sequences = []
        for trace in traces:
            if len(trace.tool_calls) >= config.mining.min_pattern_length:
                # Only include successful sequences for pattern mining
                if trace.final_success:
                    tool_sequence = [call.tool_name for call in trace.tool_calls]
                    # Limit sequence length
                    if len(tool_sequence) <= config.mining.max_pattern_length:
                        sequences.append(tool_sequence)
        
        if not sequences:
            print("⚠️  No suitable sequences found")
            return []
        
        print(f"📊 Analyzing {len(sequences)} tool sequences")
        
        # Use PrefixSpan to find frequent patterns
        ps = PrefixSpan(sequences)
        patterns = ps.frequent(min_support_count)
        
        # Filter patterns to reasonable length
        filtered_patterns = []
        for pattern, support in patterns:
            # Make sure pattern is a list and has the right length
            if isinstance(pattern, list) and config.mining.min_pattern_length <= len(pattern) <= config.mining.max_pattern_length:
                filtered_patterns.append((pattern, support))
        
        print(f"⭐ Found {len(filtered_patterns)} patterns meeting length criteria")
        return filtered_patterns
    
    def filter_skill_candidates(self, patterns: List[Tuple[List[str], int]]) -> List[SkillCandidate]:
        """Filter patterns to high-quality skill candidates using configurable thresholds."""
        candidates = []
        
        for pattern, support_count in patterns:
            # Get detailed metrics for this pattern
            metrics = asyncio.run(self._analyze_pattern_metrics(pattern))
            
            # Apply configurable filters
            if (metrics['success_rate'] >= config.mining.min_success_rate and
                metrics['cost_efficiency'] <= config.mining.max_cost_ratio):
                
                candidate = SkillCandidate(
                    tool_sequence=pattern,
                    support_count=support_count,
                    success_rate=metrics['success_rate'],
                    avg_cost=metrics['avg_cost'],
                    avg_latency_ms=metrics['avg_latency']
                )
                candidates.append(candidate)
        
        print(f"🎯 Filtered to {len(candidates)} skill candidates")
        print(f"   Filters: success_rate >= {config.mining.min_success_rate:.0%}, cost_ratio <= {config.mining.max_cost_ratio:.0%}")
        
        return candidates
    
    async def _analyze_pattern_metrics(self, pattern: List[str]) -> Dict[str, float]:
        """Analyze detailed metrics for a specific pattern."""
        
        # Get all traces that contain this pattern
        all_traces = await self.storage.get_recent_traces(limit=10000)
        
        pattern_executions = []
        baseline_costs = []
        
        for trace in all_traces:
            tool_sequence = [call.tool_name for call in trace.tool_calls]
            
            # Check if pattern appears in this trace
            if self._sequence_contains_pattern(tool_sequence, pattern):
                # Get the specific execution of this pattern
                pattern_calls = self._extract_pattern_calls(trace.tool_calls, pattern)
                if pattern_calls:
                    pattern_executions.append({
                        'success': all(call.success for call in pattern_calls),
                        'cost': sum(call.cost_estimate for call in pattern_calls),
                        'latency': sum(call.latency_ms for call in pattern_calls)
                    })
            
            # Also collect baseline costs for individual tools
            for call in trace.tool_calls:
                if call.tool_name in pattern:
                    baseline_costs.append(call.cost_estimate)
        
        if not pattern_executions:
            return {
                'success_rate': 0.0,
                'avg_cost': float('inf'),
                'avg_latency': float('inf'),
                'cost_efficiency': float('inf')
            }
        
        # Calculate metrics
        total_executions = len(pattern_executions)
        successful_executions = sum(1 for exec in pattern_executions if exec['success'])
        success_rate = successful_executions / total_executions
        
        avg_cost = sum(exec['cost'] for exec in pattern_executions) / total_executions
        avg_latency = sum(exec['latency'] for exec in pattern_executions) / total_executions
        
        # Cost efficiency: pattern cost vs sum of individual tool costs
        if baseline_costs:
            baseline_cost = sum(baseline_costs) / len(baseline_costs) * len(pattern)
            cost_efficiency = avg_cost / baseline_cost if baseline_cost > 0 else float('inf')
        else:
            cost_efficiency = 1.0
        
        return {
            'success_rate': success_rate,
            'avg_cost': avg_cost,
            'avg_latency': avg_latency,
            'cost_efficiency': cost_efficiency
        }
    
    def _sequence_contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains the pattern as a subsequence."""
        if len(pattern) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False
    
    def _extract_pattern_calls(self, tool_calls: List[ToolCall], pattern: List[str]) -> List[ToolCall]:
        """Extract the specific tool calls that match the pattern."""
        tool_sequence = [call.tool_name for call in tool_calls]
        
        for i in range(len(tool_sequence) - len(pattern) + 1):
            if tool_sequence[i:i+len(pattern)] == pattern:
                return tool_calls[i:i+len(pattern)]
        
        return []
    
    async def export_skills_csv(self, skills: List[SkillCandidate], output_path: Path):
        """Export skill candidates to CSV file."""
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow([
                'skill_id', 'tool_sequence', 'support_count', 
                'success_rate', 'avg_cost', 'avg_latency_ms', 'created_at'
            ])
            
            # Data rows
            for skill in skills:
                writer.writerow([
                    str(skill.id),
                    ' → '.join(skill.tool_sequence),
                    skill.support_count,
                    f"{skill.success_rate:.3f}",
                    f"{skill.avg_cost:.3f}",
                    f"{skill.avg_latency_ms:.1f}",
                    skill.created_at.isoformat()
                ])
        
        print(f"💾 Exported {len(skills)} skills to {output_path}")
