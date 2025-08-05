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
