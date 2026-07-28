"""Structured error records for Stage 1's extraction feedback loop.

This module was `retry_loop.py` and held a RetryLoop/RetryConfig/
RetryExhaustedError framework that CLAUDE.md documented as THE retry mechanism
for the project. Nothing outside its own test file ever imported it -- every
agent actually retries via AgentLoop (util/orchestration/loop.py) -- so the 255
dead lines were deleted and the file renamed to match what genuinely survives:
the three error-record types that Stage 1's validation middleware and
error_formatter build feedback from.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorType(Enum):
    MISSING = "missing"
    INTRODUCED = "introduced"
    CHANGED = "changed"
    DETERMINISTIC = "deterministic"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorRecord:
    iteration: int
    error_type: ErrorType
    severity: Severity
    description: str
    fact_id: Optional[int] = None
    resolved: bool = False
    signature_key: Optional[str] = None

    def signature(self) -> str:
        if self.signature_key:
            return self.signature_key
        return f"{self.error_type.value}:{self.fact_id}"


