from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, List, Any
from enum import Enum
import asyncio

T = TypeVar('T')

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

    def signature(self) -> str:
        return f"{self.error_type.value}:{self.fact_id}"

@dataclass
class RetryConfig:
    max_retries: int = 5
    filter_severities: List[Severity] = field(default_factory=lambda: [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM])
    persistent_threshold: int = 2

@dataclass
class ValidationResult(Generic[T]):
    is_valid: bool
    errors: List[ErrorRecord]
    validation_output: Optional[T] = None

class RetryLoop(Generic[T]):
    def __init__(
        self,
        agent_getter: Callable[[], Any],
        output_structure: type,
        llm_validator: Callable[[Any, Optional[str]], ValidationResult],
        error_formatter: Callable[[List[ErrorRecord], int, Optional[Any]], str],
        config: RetryConfig = None,
        deterministic_validator: Optional[Callable[[Any, str], List[ErrorRecord]]] = None
    ):
        self.agent_getter = agent_getter
        self.output_structure = output_structure
        self.llm_validator = llm_validator
        self.error_formatter = error_formatter
        self.config = config or RetryConfig()
        self.deterministic_validator = deterministic_validator

        self.error_history: List[ErrorRecord] = []
        self.iteration_count = 0
        self._last_output: Optional[Any] = None

    async def run(
        self,
        task: str,
        context: Optional[str] = None
    ) -> tuple[Any, int, List[ErrorRecord]]:
        total_tokens = 0

        while self.iteration_count < self.config.max_retries:
            self.iteration_count += 1

            agent = self.agent_getter()
            enriched_query = self._build_query(task, context)

            from src.util.invoke import get_response
            output, tokens = await get_response(
                agent=agent,
                output_structure=self.output_structure,
                query=enriched_query
            )
            total_tokens += tokens
            self._last_output = output

            deterministic_errors = []
            if self.deterministic_validator and context:
                deterministic_errors = self.deterministic_validator(output, context)
                for e in deterministic_errors:
                    e.iteration = self.iteration_count
                self.error_history.extend(deterministic_errors)

            validation_result = await self.llm_validator(output, context)
            for e in validation_result.errors:
                e.iteration = self.iteration_count
            self.error_history.extend(validation_result.errors)

            is_valid = validation_result.is_valid and len(deterministic_errors) == 0

            if is_valid:
                break

        return self._last_output, total_tokens, self.error_history

    def _build_query(self, task: str, context: Optional[str]) -> str:
        filtered_errors = self._get_active_errors()

        if not filtered_errors:
            if context:
                return f"{context}\n\n## TASK\n{task}"
            return task

        error_section = self.error_formatter(filtered_errors, self.iteration_count, self._last_output)

        if context:
            return f"{context}\n\n## ERRORS FROM PREVIOUS ATTEMPTS\n{error_section}\n\n## TASK\n{task}"
        return f"{task}\n\n## ERRORS FROM PREVIOUS ATTEMPTS\n{error_section}"

    def _get_active_errors(self) -> List[ErrorRecord]:
        active = []
        seen_signatures = {}

        for error in self.error_history:
            if error.resolved:
                continue
            if error.severity not in self.config.filter_severities:
                continue

            sig = error.signature()
            if sig in seen_signatures:
                seen_signatures[sig] += 1
                continue

            seen_signatures[sig] = 1
            active.append(error)

        for sig, count in seen_signatures.items():
            if count >= self.config.persistent_threshold:
                for e in active:
                    if e.signature() == sig:
                        e.description = f"[PERSISTENT] {e.description}"

        return active

    def get_error_summary(self) -> dict:
        by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        by_type = {t.value: 0 for t in ErrorType}

        for e in self.error_history:
            if e.severity in by_severity:
                by_severity[e.severity] += 1
            by_type[e.error_type.value] += 1

        return {
            "total_errors": len(self.error_history),
            "by_severity": by_severity,
            "by_type": by_type,
            "iterations": self.iteration_count
        }
