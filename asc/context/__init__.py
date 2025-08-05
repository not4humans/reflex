"""
Context Analysis Module for Context-Aware Skill Learning

Analyzes execution traces to identify context patterns that predict success/failure.
This enables skills to be context-sensitive like human procedural learning.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import json
from pathlib import Path

from ..core.models import TaskTrace, ToolCall
from ..storage.traces import TraceStorage


async def analyze_context_patterns(traces: List[TaskTrace]) -> Dict[str, Any]:
    """
    Analyze execution traces to extract context patterns.
    
    Returns patterns for success conditions, failure indicators, and adaptations.
    """
    
    if not traces:
        return {
            'success_conditions': {},
            'failure_patterns': {},
            'context_adaptations': {}
        }
    
    success_contexts = []
    failure_contexts = []
    
    # Separate successful and failed executions
    for trace in traces:
        if hasattr(trace, 'final_success') and trace.final_success:
            success_contexts.append(_extract_context_from_trace(trace))
        else:
            failure_contexts.append(_extract_context_from_trace(trace))
    
    # Analyze patterns
    success_conditions = _analyze_success_conditions(success_contexts)
    failure_patterns = _analyze_failure_patterns(failure_contexts, success_contexts)
    context_adaptations = _analyze_context_adaptations(traces)
    
    return {
        'success_conditions': success_conditions,
        'failure_patterns': failure_patterns,
        'context_adaptations': context_adaptations
    }


def _extract_context_from_trace(trace: TaskTrace) -> Dict[str, Any]:
    """Extract context information from a single trace."""
    
    context = {}
    
    # Extract from tool calls if they have context information
    for call in trace.tool_calls:
        if hasattr(call, 'execution_context') and call.execution_context:
            context.update(call.execution_context)
        
        # Infer context from tool patterns
        if call.tool_name == 'azure_login':
            context['requires_authentication'] = True
        elif call.tool_name == 'create_resource_group':
            context['resource_group_missing'] = True
        elif 'network' in call.tool_name.lower() or 'timeout' in str(call.error or '').lower():
            context['network_issues'] = True
        elif 'disk' in str(call.error or '').lower() or 'space' in str(call.error or '').lower():
            context['disk_space_issues'] = True
    
    # Add trace-level context
    context['total_tool_calls'] = len(trace.tool_calls)
    context['trace_duration'] = getattr(trace, 'duration_ms', 0)
    context['has_errors'] = any(call.error for call in trace.tool_calls)
    
    return context


def _analyze_success_conditions(success_contexts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Analyze what conditions are commonly present in successful executions."""
    
    if not success_contexts:
        return {}
    
    condition_counts = defaultdict(int)
    total_successes = len(success_contexts)
    
    # Count frequency of conditions in successful executions
    for context in success_contexts:
        for key, value in context.items():
            if isinstance(value, bool) and value:
                condition_counts[f"{key}=true"] += 1
            elif isinstance(value, (int, float, str)) and value:
                condition_counts[f"{key}={value}"] += 1
    
    # Filter to conditions that appear in at least 60% of successes
    threshold = max(1, int(total_successes * 0.6))
    success_conditions = {}
    
    for condition, count in condition_counts.items():
        if count >= threshold:
            category = _categorize_condition(condition)
            if category not in success_conditions:
                success_conditions[category] = []
            success_conditions[category].append(condition)
    
    return success_conditions


def _analyze_failure_patterns(failure_contexts: List[Dict[str, Any]], 
                            success_contexts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Analyze what conditions are associated with failures."""
    
    if not failure_contexts:
        return {}
    
    failure_conditions = defaultdict(int)
    success_conditions = defaultdict(int)
    
    total_failures = len(failure_contexts)
    total_successes = len(success_contexts)
    
    # Count conditions in failures
    for context in failure_contexts:
        for key, value in context.items():
            if isinstance(value, bool) and value:
                failure_conditions[f"{key}=true"] += 1
            elif isinstance(value, (int, float, str)) and value:
                failure_conditions[f"{key}={value}"] += 1
    
    # Count conditions in successes for comparison
    for context in success_contexts:
        for key, value in context.items():
            if isinstance(value, bool) and value:
                success_conditions[f"{key}=true"] += 1
            elif isinstance(value, (int, float, str)) and value:
                success_conditions[f"{key}={value}"] += 1
    
    # Calculate failure patterns
    failure_patterns = {}
    
    for condition, failure_count in failure_conditions.items():
        if failure_count >= 2:  # At least 2 failures with this condition
            success_count = success_conditions.get(condition, 0)
            
            # Calculate failure rate for this condition
            total_with_condition = failure_count + success_count
            if total_with_condition > 0:
                failure_rate = failure_count / total_with_condition
                
                # Only include if failure rate is significantly higher than baseline
                baseline_failure_rate = total_failures / (total_failures + total_successes) if (total_failures + total_successes) > 0 else 0.5
                
                if failure_rate > baseline_failure_rate * 1.5:  # 50% higher than baseline
                    confidence = min(0.95, failure_count / max(1, total_failures))
                    failure_patterns[condition] = {
                        'failure_rate': failure_rate,
                        'confidence': confidence
                    }
    
    return failure_patterns


def _analyze_context_adaptations(traces: List[TaskTrace]) -> Dict[str, str]:
    """Analyze what adaptations are commonly applied in different contexts."""
    
    adaptations = {}
    
    # Common adaptation patterns based on tool usage patterns
    adaptation_patterns = {
        'if_not_authenticated': 'force_login_with_device_code',
        'if_resource_group_missing': 'create_resource_group_first',
        'if_path_invalid': 'validate_and_fix_path',
        'if_network_unstable': 'retry_with_exponential_backoff',
        'if_insufficient_permissions': 'escalate_or_use_alternate_approach',
        'if_disk_space_low': 'cleanup_temp_files_first'
    }
    
    # Analyze traces for evidence of these patterns
    for trace in traces:
        tool_sequence = [call.tool_name for call in trace.tool_calls]
        
        # Look for authentication patterns
        if 'azure_login' in tool_sequence or 'auth' in ' '.join(tool_sequence).lower():
            adaptations['if_not_authenticated'] = 'force_login_with_device_code'
        
        # Look for resource creation patterns
        if 'create_resource_group' in tool_sequence or 'create' in ' '.join(tool_sequence).lower():
            adaptations['if_resource_group_missing'] = 'create_resource_group_first'
        
        # Look for retry patterns
        if len([call for call in trace.tool_calls if call.tool_name == trace.tool_calls[0].tool_name]) > 1:
            adaptations['if_network_unstable'] = 'retry_with_exponential_backoff'
    
    # Add default adaptations if not found
    adaptations.update(adaptation_patterns)
    
    return adaptations


def _categorize_condition(condition: str) -> str:
    """Categorize a condition into a logical group."""
    
    condition_lower = condition.lower()
    
    if 'auth' in condition_lower or 'login' in condition_lower:
        return 'authentication'
    elif 'resource' in condition_lower or 'group' in condition_lower:
        return 'prerequisites'
    elif 'network' in condition_lower or 'connection' in condition_lower:
        return 'connectivity'
    elif 'disk' in condition_lower or 'space' in condition_lower or 'memory' in condition_lower:
        return 'resources'
    elif 'path' in condition_lower or 'file' in condition_lower:
        return 'filesystem'
    else:
        return 'general'


class ContextAnalyzer:
    """Analyzes execution contexts to identify success/failure patterns."""
    
    def __init__(self, storage: TraceStorage):
        self.storage = storage
        
    async def analyze_pattern_contexts(self, tool_sequence: List[str], 
                                     min_traces: int = 5) -> Dict[str, Any]:
        """
        Analyze context patterns for a specific tool sequence.
        
        Returns context conditions that predict success/failure.
        """
        # Get all traces containing this pattern
        matching_traces = await self._find_pattern_traces(tool_sequence)
        
        if len(matching_traces) < min_traces:
            return {"insufficient_data": True, "trace_count": len(matching_traces)}
        
        # Separate successful and failed executions
        successful_traces = [t for t in matching_traces if t.final_success]
        failed_traces = [t for t in matching_traces if not t.final_success]
        
        if not successful_traces:
            return {"no_successful_executions": True}
        
        # Analyze context patterns
        success_contexts = self._extract_contexts(successful_traces, tool_sequence)
        failure_contexts = self._extract_contexts(failed_traces, tool_sequence) if failed_traces else []
        
        # Identify success predictors
        success_conditions = self._find_success_conditions(success_contexts, failure_contexts)
        
        # Identify failure patterns
        failure_patterns = self._find_failure_patterns(failure_contexts, success_contexts)
        
        # Generate context adaptations
        adaptations = self._generate_adaptations(success_conditions, failure_patterns)
        
        return {
            "success_conditions": success_conditions,
            "failure_patterns": failure_patterns,
            "context_adaptations": adaptations,
            "analysis_stats": {
                "total_traces": len(matching_traces),
                "successful_traces": len(successful_traces),
                "failed_traces": len(failed_traces),
                "success_rate": len(successful_traces) / len(matching_traces)
            }
        }
    
    async def _find_pattern_traces(self, tool_sequence: List[str]) -> List[TaskTrace]:
        """Find all traces that contain the specified tool sequence."""
        all_traces = await self.storage.get_all_traces()
        matching_traces = []
        
        for trace in all_traces:
            trace_tools = [call.tool_name for call in trace.tool_calls]
            if self._contains_sequence(trace_tools, tool_sequence):
                matching_traces.append(trace)
        
        return matching_traces
    
    def _contains_sequence(self, tools: List[str], sequence: List[str]) -> bool:
        """Check if tools list contains the sequence."""
        if len(sequence) > len(tools):
            return False
        
        for i in range(len(tools) - len(sequence) + 1):
            if tools[i:i+len(sequence)] == sequence:
                return True
        return False
    
    def _extract_contexts(self, traces: List[TaskTrace], tool_sequence: List[str]) -> List[Dict[str, Any]]:
        """Extract execution contexts from traces for the specified pattern."""
        contexts = []
        
        for trace in traces:
            # Find where the pattern occurs in the trace
            trace_tools = [call.tool_name for call in trace.tool_calls]
            
            for i in range(len(trace_tools) - len(tool_sequence) + 1):
                if trace_tools[i:i+len(tool_sequence)] == tool_sequence:
                    # Extract context from tool calls in this pattern
                    pattern_calls = trace.tool_calls[i:i+len(tool_sequence)]
                    context = self._build_context_from_calls(pattern_calls, trace)
                    contexts.append(context)
                    break  # Only analyze first occurrence in each trace
        
        return contexts
    
    def _build_context_from_calls(self, tool_calls: List[ToolCall], trace: TaskTrace) -> Dict[str, Any]:
        """Build context dictionary from tool calls and trace."""
        context = {
            "task_type": self._categorize_task(trace.task_description),
            "sequence_length": len(tool_calls),
            "total_trace_length": len(trace.tool_calls),
            "time_of_day": tool_calls[0].timestamp.hour if tool_calls else 0,
        }
        
        # Extract context from individual tool calls
        for i, call in enumerate(tool_calls):
            call_context = call.execution_context or {}
            preconditions = call.preconditions_met or {}
            
            # Merge contexts with prefixes to avoid conflicts
            for key, value in call_context.items():
                context[f"step_{i}_{key}"] = value
            
            for key, value in preconditions.items():
                context[f"precond_{key}"] = value
        
        # Extract common argument patterns
        context.update(self._extract_argument_patterns(tool_calls))
        
        return context
    
    def _categorize_task(self, task_description: str) -> str:
        """Categorize task type based on description."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["deploy", "azure", "cloud", "server"]):
            return "deployment"
        elif any(word in task_lower for word in ["text", "transform", "string", "file"]):
            return "text_processing"
        elif any(word in task_lower for word in ["math", "calculate", "number", "compute"]):
            return "computation"
        elif any(word in task_lower for word in ["data", "analyze", "process", "extract"]):
            return "data_processing"
        else:
            return "general"
    
    def _extract_argument_patterns(self, tool_calls: List[ToolCall]) -> Dict[str, Any]:
        """Extract common patterns from tool arguments."""
        patterns = {}
        
        # Look for common argument keys
        all_args = {}
        for call in tool_calls:
            all_args.update(call.args)
        
        # Categorize argument types
        for key, value in all_args.items():
            if isinstance(value, str):
                if "path" in key.lower() or "file" in key.lower():
                    patterns[f"has_{key}"] = len(value) > 0
                    patterns[f"{key}_type"] = "absolute" if value.startswith("/") else "relative"
                elif "url" in key.lower():
                    patterns[f"has_{key}"] = len(value) > 0
                    patterns[f"{key}_protocol"] = value.split(":")[0] if ":" in value else "unknown"
        
        return patterns
    
    def _find_success_conditions(self, success_contexts: List[Dict[str, Any]], 
                               failure_contexts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find context conditions that predict success."""
        if not success_contexts:
            return {}
        
        # Count occurrences in successful vs failed contexts
        success_counter = Counter()
        failure_counter = Counter()
        
        for context in success_contexts:
            for key, value in context.items():
                if isinstance(value, (bool, str, int, float)):
                    condition = f"{key}={value}"
                    success_counter[condition] += 1
        
        for context in failure_contexts:
            for key, value in context.items():
                if isinstance(value, (bool, str, int, float)):
                    condition = f"{key}={value}"
                    failure_counter[condition] += 1
        
        # Find conditions that are more common in success
        total_success = len(success_contexts)
        total_failure = len(failure_contexts) or 1  # Avoid division by zero
        
        success_conditions = defaultdict(list)
        
        for condition, success_count in success_counter.items():
            failure_count = failure_counter.get(condition, 0)
            
            success_rate = success_count / total_success
            failure_rate = failure_count / total_failure
            
            # Condition is a good predictor if it appears in >50% of successes
            # and significantly less in failures
            if success_rate > 0.5 and success_rate > failure_rate * 1.5:
                category = self._categorize_condition(condition)
                success_conditions[category].append(condition)
        
        return dict(success_conditions)
    
    def _find_failure_patterns(self, failure_contexts: List[Dict[str, Any]], 
                             success_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find context patterns that predict failure."""
        if not failure_contexts:
            return {}
        
        failure_counter = Counter()
        success_counter = Counter()
        
        for context in failure_contexts:
            for key, value in context.items():
                if isinstance(value, (bool, str, int, float)):
                    condition = f"{key}={value}"
                    failure_counter[condition] += 1
        
        for context in success_contexts:
            for key, value in context.items():
                if isinstance(value, (bool, str, int, float)):
                    condition = f"{key}={value}"
                    success_counter[condition] += 1
        
        total_failure = len(failure_contexts)
        total_success = len(success_contexts) or 1
        
        failure_patterns = {}
        
        for condition, failure_count in failure_counter.items():
            success_count = success_counter.get(condition, 0)
            
            failure_rate = failure_count / total_failure
            success_rate = success_count / total_success
            
            # Pattern is a failure predictor if it appears in >50% of failures
            # and significantly less in successes
            if failure_rate > 0.5 and failure_rate > success_rate * 1.5:
                failure_patterns[condition] = {
                    "failure_rate": failure_rate,
                    "success_rate": success_rate,
                    "confidence": failure_rate - success_rate
                }
        
        return failure_patterns
    
    def _categorize_condition(self, condition: str) -> str:
        """Categorize a condition into logical groups."""
        condition_lower = condition.lower()
        
        if "precond_" in condition_lower:
            return "prerequisites"
        elif "path" in condition_lower or "file" in condition_lower:
            return "file_system"
        elif "auth" in condition_lower or "login" in condition_lower:
            return "authentication"
        elif "network" in condition_lower or "connection" in condition_lower:
            return "connectivity"
        elif "resource" in condition_lower or "space" in condition_lower:
            return "resources"
        else:
            return "environment"
    
    def _generate_adaptations(self, success_conditions: Dict[str, List[str]], 
                            failure_patterns: Dict[str, Any]) -> Dict[str, str]:
        """Generate adaptation rules based on context analysis."""
        adaptations = {}
        
        # Generate adaptations for missing success conditions
        for category, conditions in success_conditions.items():
            if category == "authentication":
                adaptations["if_not_authenticated"] = "force_login_with_retry"
            elif category == "prerequisites":
                adaptations["if_prerequisites_missing"] = "validate_and_setup_prerequisites"
            elif category == "file_system":
                adaptations["if_path_invalid"] = "validate_and_create_path"
            elif category == "resources":
                adaptations["if_insufficient_resources"] = "check_and_cleanup_resources"
        
        # Generate adaptations for known failure patterns
        for condition, pattern_info in failure_patterns.items():
            if "network" in condition.lower():
                adaptations["if_network_unstable"] = "retry_with_backoff"
            elif "space" in condition.lower():
                adaptations["if_disk_space_low"] = "cleanup_temp_files_first"
            elif "permission" in condition.lower():
                adaptations["if_permission_denied"] = "escalate_permissions_or_alternate_path"
        
        return adaptations


async def enhance_skill_with_context(skill_metadata: Dict[str, Any], 
                                   tool_sequence: List[str],
                                   storage: TraceStorage) -> Dict[str, Any]:
    """Enhance existing skill metadata with context analysis."""
    analyzer = ContextAnalyzer(storage)
    context_analysis = await analyzer.analyze_pattern_contexts(tool_sequence)
    
    if context_analysis.get("insufficient_data"):
        # Not enough data for context analysis
        return skill_metadata
    
    # Add context information to skill metadata
    enhanced_metadata = skill_metadata.copy()
    enhanced_metadata.update({
        "success_conditions": context_analysis.get("success_conditions", {}),
        "context_adaptations": context_analysis.get("context_adaptations", {}),
        "failure_patterns": context_analysis.get("failure_patterns", {}),
        "context_analysis_stats": context_analysis.get("analysis_stats", {})
    })
    
    return enhanced_metadata
