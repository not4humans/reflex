"""Core data models for trace collection and skill compilation."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation with all metadata."""
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    session_id: UUID
    agent_id: str
    
    # Tool execution details
    tool_name: str
    args: Dict[str, Any]
    result: Any
    success: bool
    error_message: Optional[str] = None
    
    # Cost and performance metrics
    latency_ms: float
    cost_estimate: float  # In arbitrary cost units for now
    
    # Context for pattern mining
    context_embedding: Optional[List[float]] = None  # Will implement later
    preceding_tools: List[str] = Field(default_factory=list)  # Last N tool names


class TaskTrace(BaseModel):
    """A complete trace of task execution."""
    
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    task_description: str
    
    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: Optional[datetime] = None
    
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_success: Optional[bool] = None
    total_cost: float = 0.0
    total_latency_ms: float = 0.0


class SkillCandidate(BaseModel):
    """A potential skill identified through pattern mining."""
    
    id: UUID = Field(default_factory=uuid4)
    tool_sequence: List[str]
    support_count: int  # How many times this pattern appeared
    success_rate: float  # Percentage of successful executions
    avg_cost: float
    avg_latency_ms: float
    
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    validation_status: Optional[str] = None  # "pending", "passed", "failed"


class ToolPattern(BaseModel):
    """A discovered pattern of tool usage for skill compilation."""
    
    pattern_id: str
    tool_sequence: List[str]
    frequency: int
    avg_success_rate: float
    avg_cost: float
    avg_latency_ms: float
    example_traces: List[str]  # Session IDs of example traces
    task_types: List[str]  # Common task descriptions
    discovered_at: datetime = Field(default_factory=lambda: datetime.now())


class CompiledSkillMetadata(BaseModel):
    """Metadata for a compiled skill."""
    
    skill_id: str
    name: str
    description: str
    pattern: List[str]  # Original tool sequence
    parameters: List[str]  # Skill function parameters
    strategy: str  # Compilation strategy used
    confidence: float  # Compilation confidence
    estimated_cost_reduction: float
    success_rate: float
    avg_cost: float
    avg_latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    source_pattern_id: Optional[str] = None


class SkillRetrievalCandidate(BaseModel):
    """A candidate skill for task execution (Phase 6)."""
    
    skill_id: str
    skill_name: str
    similarity_score: float  # Semantic similarity to task
    confidence_score: float  # Skill compilation confidence
    combined_score: float    # Weighted combination
    metadata: CompiledSkillMetadata
    
    def meets_threshold(self, threshold: float) -> bool:
        """Check if this candidate meets the confidence threshold."""
        return self.combined_score >= threshold


class ValidationResult(BaseModel):
    """Results from skill validation testing."""
    
    skill_id: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    avg_execution_time_ms: float
    total_cost: float
    error_types: Dict[str, int]  # Error type -> count
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for skill-aware agents."""
    
    agent_id: str
    total_tasks: int = 0
    skill_executions: int = 0
    fallback_executions: int = 0
    skill_success_rate: float = 0.0
    fallback_success_rate: float = 0.0
    avg_skill_latency_ms: float = 0.0
    avg_fallback_latency_ms: float = 0.0
    avg_skill_cost: float = 0.0
    avg_fallback_cost: float = 0.0
    cost_savings: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    
    def calculate_skill_usage_rate(self) -> float:
        """Calculate the rate of skill usage vs fallback."""
        if self.total_tasks == 0:
            return 0.0
        return self.skill_executions / self.total_tasks
    
    def calculate_cost_savings(self) -> float:
        """Calculate cost savings from using skills."""
        if self.avg_fallback_cost == 0:
            return 0.0
        return (self.avg_fallback_cost - self.avg_skill_cost) / self.avg_fallback_cost
