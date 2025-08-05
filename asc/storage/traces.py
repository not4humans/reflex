"""SQLite-based trace storage for development and experimentation."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aiosqlite

from ..core.models import TaskTrace, ToolCall, SkillCandidate


class TraceStorage:
    """Async SQLite storage for traces and skill candidates."""
    
    def __init__(self, db_path: Path = Path("data/traces.db")):
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()  # Prevent concurrent database operations
    
    async def initialize(self):
        """Create tables if they don't exist and set up WAL mode."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Set WAL mode once during initialization
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL
                await db.execute("PRAGMA cache_size=10000")   # Larger cache
                await db.execute("PRAGMA temp_store=memory")  # Use memory for temp tables
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS task_traces (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        task_description TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        final_success BOOLEAN,
                        total_cost REAL NOT NULL DEFAULT 0,
                        total_latency_ms REAL NOT NULL DEFAULT 0
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS tool_calls (
                        id TEXT PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        session_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        args TEXT NOT NULL,  -- JSON
                        result TEXT,         -- JSON
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        latency_ms REAL NOT NULL,
                        cost_estimate REAL NOT NULL,
                        context_embedding TEXT,  -- JSON array
                        preceding_tools TEXT     -- JSON array
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS skill_candidates (
                        id TEXT PRIMARY KEY,
                        tool_sequence TEXT NOT NULL,  -- JSON array
                        support_count INTEGER NOT NULL,
                        success_rate REAL NOT NULL,
                        avg_cost REAL NOT NULL,
                        avg_latency_ms REAL NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        validation_status TEXT
                    )
                """)
                
                await db.commit()
    
    async def store_tool_call(self, tool_call: ToolCall):
        """Store a single tool call."""
        async with self._lock:  # Serialize database access
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO tool_calls (
                        id, timestamp, session_id, agent_id, tool_name, args, result,
                        success, error_message, latency_ms, cost_estimate,
                        context_embedding, preceding_tools
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(tool_call.id),
                    tool_call.timestamp.isoformat(),
                    str(tool_call.session_id),
                    tool_call.agent_id,
                    tool_call.tool_name,
                    json.dumps(tool_call.args),
                    json.dumps(tool_call.result),
                    tool_call.success,
                    tool_call.error_message,
                    tool_call.latency_ms,
                    tool_call.cost_estimate,
                    json.dumps(tool_call.context_embedding) if tool_call.context_embedding else None,
                    json.dumps(tool_call.preceding_tools)
                ))
                await db.commit()
    
    async def store_task_trace(self, trace: TaskTrace):
        """Store a complete task trace."""
        async with self._lock:  # Serialize database access
            async with aiosqlite.connect(self.db_path) as db:
                # Store the main trace record
                await db.execute("""
                    INSERT OR REPLACE INTO task_traces (
                        id, session_id, agent_id, task_description,
                        start_time, end_time, final_success, total_cost, total_latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(trace.id),
                    str(trace.session_id),
                    trace.agent_id,
                    trace.task_description,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.final_success,
                    trace.total_cost,
                    trace.total_latency_ms
                ))
                
                # Note: tool calls are stored individually via store_tool_call
                await db.commit()
    
    async def get_recent_traces(self, limit: int = 100) -> List[TaskTrace]:
        """Get recent task traces for pattern mining."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Get task traces
                async with db.execute("""
                    SELECT * FROM task_traces 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (limit,)) as cursor:
                    trace_rows = await cursor.fetchall()
                
                traces = []
                for trace_row in trace_rows:
                    # Get tool calls for this trace
                    async with db.execute("""
                        SELECT * FROM tool_calls 
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """, (trace_row['session_id'],)) as tool_cursor:
                        tool_rows = await tool_cursor.fetchall()
                    
                    # Build tool calls
                    tool_calls = []
                    for tool_row in tool_rows:
                        tool_call = ToolCall(
                            id=UUID(tool_row['id']),
                            timestamp=datetime.fromisoformat(tool_row['timestamp']),
                            session_id=UUID(tool_row['session_id']),
                            agent_id=tool_row['agent_id'],
                            tool_name=tool_row['tool_name'],
                            args=json.loads(tool_row['args']),
                            result=tool_row['result'],
                            success=bool(tool_row['success']),
                            error_message=tool_row['error_message'],
                            latency_ms=tool_row['latency_ms'],
                            cost_estimate=tool_row['cost_estimate'],
                            context_embedding=json.loads(tool_row['context_embedding']) if tool_row['context_embedding'] else None,
                            preceding_tools=json.loads(tool_row['preceding_tools']) if tool_row['preceding_tools'] else []
                        )
                        tool_calls.append(tool_call)
                    
                    # Build task trace
                    trace = TaskTrace(
                        id=UUID(trace_row['id']),
                        session_id=UUID(trace_row['session_id']),
                        agent_id=trace_row['agent_id'],
                        task_description=trace_row['task_description'],
                        start_time=datetime.fromisoformat(trace_row['start_time']),
                        end_time=datetime.fromisoformat(trace_row['end_time']) if trace_row['end_time'] else None,
                        tool_calls=tool_calls,
                        final_success=bool(trace_row['final_success']) if trace_row['final_success'] is not None else None,
                        total_cost=trace_row['total_cost'],
                        total_latency_ms=trace_row['total_latency_ms']
                    )
                    traces.append(trace)
                
                return traces
    
    async def get_trace_count(self) -> int:
        """Get total number of stored traces."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM task_traces")
                result = await cursor.fetchone()
                return result[0] if result else 0
    
    async def clear_all_traces(self):
        """Clear all traces from the database (for testing)."""
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM tool_calls")
                await db.execute("DELETE FROM task_traces")
                await db.commit()
