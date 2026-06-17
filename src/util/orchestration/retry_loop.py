from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Optional, List, Set
from enum import Enum

from pydantic import BaseModel

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
    signature_key: Optional[str] = None

    def signature(self) -> str:
        if self.signature_key:
            return self.signature_key
        return f"{self.error_type.value}:{self.fact_id}"


class RetryExhaustedError(Exception):
    def __init__(self, errors: List[ErrorRecord], last_output: Optional[object], total_tokens: int):
        self.errors = errors
        self.last_output = last_output
        self.total_tokens = total_tokens
        super().__init__(
            f"Retry loop exhausted with {len(errors)} unresolved serious error(s)."
        )

@dataclass
class RetryConfig:
    max_retries: int = 5
    filter_severities: List[Severity] = field(default_factory=lambda: [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM])
    persistent_threshold: int = 2
    accumulate_accepted: bool = False
    serialize_accepted: Optional[Callable[[object], str]] = None

@dataclass
class ValidationResult(Generic[T]):
    is_valid: bool
    errors: List[ErrorRecord]
    validation_output: Optional[T] = None
    token_usage: int = 0


class RetryErrorSummary(BaseModel):
    total_errors: int
    critical: int
    high: int
    medium: int
    low: int
    missing: int
    introduced: int
    changed: int
    deterministic: int
    iterations: int

class RetryLoop(Generic[T]):
    def __init__(
        self,
        agent_getter: Callable[[], object],
        output_structure: type,
        llm_validator: Callable[[object, Optional[str]], ValidationResult],
        error_formatter: Callable[[List[ErrorRecord], int, Optional[object]], str],
        config: Optional[RetryConfig] = None,
        deterministic_validator: Optional[Callable[[object, str], List[ErrorRecord]]] = None,
    ):
        self.agent_getter = agent_getter
        self.output_structure = output_structure
        self.llm_validator = llm_validator
        self.error_formatter = error_formatter
        self.config = config or RetryConfig()
        self.deterministic_validator = deterministic_validator

        self.error_history: List[ErrorRecord] = []
        self.iteration_count = 0
        self._last_output: Optional[object] = None
        self._last_is_valid = False
        self._latest_errors: List[ErrorRecord] = []
        self._accepted_fact_ids: Set[int] = set()

    def _update_accepted_fact_ids(self, iteration_errors: List[ErrorRecord]) -> None:
        """Update the set of fact IDs that have no errors across all iterations."""
        if not self.config.accumulate_accepted:
            return
        errored_ids: Set[int] = {e.fact_id for e in self.error_history if e.fact_id is not None}
        all_fact_ids: Set[int] = set()
        # Collect all fact IDs from the current output
        if self._last_output and hasattr(self._last_output, 'facts'):
            all_fact_ids = {getattr(f, 'id', None) for f in self._last_output.facts if getattr(f, 'id', None) is not None}
        self._accepted_fact_ids = all_fact_ids - errored_ids

    def _format_accepted_facts(self) -> Optional[str]:
        """Format the accepted facts section for the query prompt."""
        if not self.config.accumulate_accepted:
            return None
        if not self._accepted_fact_ids or self._last_output is None:
            return None
        if self.config.serialize_accepted:
            return self.config.serialize_accepted(self._last_output)
        # Default serialization: extract facts by accepted IDs if output has 'facts'
        if hasattr(self._last_output, 'facts'):
            lines = []
            for fact in self._last_output.facts:
                fid = getattr(fact, 'id', None)
                if fid is not None and fid in self._accepted_fact_ids:
                    f_text = getattr(fact, 'fact', str(fact))
                    lines.append(f"- id: {fid}\n  fact: {f_text}")
            if lines:
                return "\n".join(lines)
        return None

    async def run(
        self,
        task: str,
        context: Optional[str] = None
    ) -> tuple[object, int, List[ErrorRecord]]:
        total_tokens = 0

        while self.iteration_count < self.config.max_retries:
            self.iteration_count += 1

            agent = self.agent_getter()
            enriched_query = self._build_query(task, context)

            from src.util.core.invoke import get_response
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
            total_tokens += validation_result.token_usage
            self.error_history.extend(validation_result.errors)

            iteration_errors = deterministic_errors + validation_result.errors
            self._latest_errors = iteration_errors
            self._update_accepted_fact_ids(iteration_errors)
            is_valid = validation_result.is_valid and len(deterministic_errors) == 0

            if is_valid:
                self._last_is_valid = True
                break

        active_errors = self._get_latest_blocking_errors()
        if not self._last_is_valid and active_errors:
            raise RetryExhaustedError(active_errors, self._last_output, total_tokens)

        return self._last_output, total_tokens, self.error_history

    def _build_query(self, task: str, context: Optional[str]) -> str:
        filtered_errors = self._get_prompt_errors()
        accepted_section = self._format_accepted_facts()

        parts: List[str] = []
        if context:
            parts.append(context)

        if accepted_section:
            parts.append(f"## ACCEPTED FACTS (keep these unchanged)\n{accepted_section}\nThese facts have passed all previous validation checks. Do NOT regenerate, reword, or remove them. Include them verbatim in your output.")

        if filtered_errors:
            error_section = self.error_formatter(filtered_errors, self.iteration_count, self._last_output)
            parts.append(f"## ERRORS FROM PREVIOUS ATTEMPTS\n{error_section}")

        parts.append(f"## TASK\n{task}")

        return "\n\n".join(parts)

    def _get_prompt_errors(self) -> List[ErrorRecord]:
        active: List[ErrorRecord] = []
        seen_signatures: List[str] = []

        for error in self._latest_errors:
            if error.resolved:
                continue
            if error.severity not in self.config.filter_severities:
                continue

            sig = error.signature()
            if sig in seen_signatures:
                continue

            seen_signatures.append(sig)
            active.append(self._with_persistent_marker(error))

        return active

    def _get_latest_blocking_errors(self) -> List[ErrorRecord]:
        active: List[ErrorRecord] = []
        seen_signatures: List[str] = []
        for error in self._latest_errors:
            if error.resolved:
                continue
            if error.severity not in self.config.filter_severities:
                continue
            sig = error.signature()
            if sig in seen_signatures:
                continue
            seen_signatures.append(sig)
            active.append(error)
        return active

    def _with_persistent_marker(self, error: ErrorRecord) -> ErrorRecord:
        occurrences = sum(1 for candidate in self.error_history if candidate.signature() == error.signature())
        if occurrences < self.config.persistent_threshold or error.description.startswith("[PERSISTENT]"):
            return error
        return ErrorRecord(
            iteration=error.iteration,
            error_type=error.error_type,
            severity=error.severity,
            description=f"[PERSISTENT] {error.description}",
            fact_id=error.fact_id,
            resolved=error.resolved,
            signature_key=error.signature_key,
        )

    def get_error_summary(self) -> RetryErrorSummary:
        critical = 0
        high = 0
        medium = 0
        low = 0
        missing = 0
        introduced = 0
        changed = 0
        deterministic = 0

        for e in self.error_history:
            if e.severity == Severity.CRITICAL:
                critical += 1
            elif e.severity == Severity.HIGH:
                high += 1
            elif e.severity == Severity.MEDIUM:
                medium += 1
            elif e.severity == Severity.LOW:
                low += 1

            if e.error_type == ErrorType.MISSING:
                missing += 1
            elif e.error_type == ErrorType.INTRODUCED:
                introduced += 1
            elif e.error_type == ErrorType.CHANGED:
                changed += 1
            elif e.error_type == ErrorType.DETERMINISTIC:
                deterministic += 1

        return RetryErrorSummary(
            total_errors=len(self.error_history),
            critical=critical,
            high=high,
            medium=medium,
            low=low,
            missing=missing,
            introduced=introduced,
            changed=changed,
            deterministic=deterministic,
            iterations=self.iteration_count,
        )
