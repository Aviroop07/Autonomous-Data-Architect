from __future__ import annotations

import asyncio
import pytest
from pydantic import BaseModel

from src.util.orchestration.retry_loop import (
    ErrorRecord,
    ErrorType,
    RetryConfig,
    RetryExhaustedError,
    RetryLoop,
    Severity,
    ValidationResult,
)


class SimpleOutput(BaseModel):
    value: str


class FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, payload: object):
        self.calls += 1
        return {
            "structured_response": SimpleOutput(value=f"attempt {self.calls}"),
            "messages": [],
        }


async def _always_invalid_validator(output: object, context: str | None):
    return ValidationResult(
        is_valid=False,
        errors=[ErrorRecord(
            iteration=0,
            error_type=ErrorType.MISSING,
            severity=Severity.MEDIUM,
            description="Missing relationship fact.",
            signature_key="missing_relationship:a:b",
        )],
    )


def _format_errors(errors: list[ErrorRecord], iteration: int, output: object) -> str:
    return "\n".join(error.description for error in errors)


def test_retry_loop_raises_when_serious_errors_remain_after_max_retries():
    loop = RetryLoop(
        agent_getter=lambda: FakeAgent(),
        output_structure=SimpleOutput,
        llm_validator=_always_invalid_validator,
        error_formatter=_format_errors,
        config=RetryConfig(max_retries=2),
    )

    with pytest.raises(RetryExhaustedError) as exc_info:
        asyncio.run(loop.run(task="test", context="context"))

    assert len(exc_info.value.errors) == 1
    assert exc_info.value.errors[0].signature() == "missing_relationship:a:b"
    assert isinstance(exc_info.value.last_output, SimpleOutput)


async def _low_only_validator(output: object, context: str | None):
    return ValidationResult(
        is_valid=False,
        errors=[ErrorRecord(
            iteration=0,
            error_type=ErrorType.MISSING,
            severity=Severity.LOW,
            description="Low-only warning.",
        )],
    )


def test_retry_loop_does_not_raise_for_only_low_severity_errors():
    loop = RetryLoop(
        agent_getter=lambda: FakeAgent(),
        output_structure=SimpleOutput,
        llm_validator=_low_only_validator,
        error_formatter=_format_errors,
        config=RetryConfig(max_retries=1),
    )

    output, _tokens, errors = asyncio.run(loop.run(task="test", context="context"))

    assert isinstance(output, SimpleOutput)
    assert len(errors) == 1


def test_retry_loop_raises_only_latest_unresolved_blocking_errors():
    call_count = 0

    async def validator(output: object, context: str | None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ValidationResult(
                is_valid=False,
                errors=[ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.MISSING,
                    severity=Severity.MEDIUM,
                    description="Old error.",
                    signature_key="old:error",
                )],
            )
        return ValidationResult(
            is_valid=False,
            errors=[ErrorRecord(
                iteration=0,
                error_type=ErrorType.MISSING,
                severity=Severity.MEDIUM,
                description="New error.",
                signature_key="new:error",
            )],
        )

    loop = RetryLoop(
        agent_getter=lambda: FakeAgent(),
        output_structure=SimpleOutput,
        llm_validator=validator,
        error_formatter=_format_errors,
        config=RetryConfig(max_retries=2),
    )

    with pytest.raises(RetryExhaustedError) as exc_info:
        asyncio.run(loop.run(task="test", context="context"))

    assert [error.signature() for error in exc_info.value.errors] == ["new:error"]


def test_retry_loop_counts_validator_token_usage():
    async def validator(output: object, context: str | None):
        return ValidationResult(is_valid=True, errors=[], token_usage=7)

    loop = RetryLoop(
        agent_getter=lambda: FakeAgent(),
        output_structure=SimpleOutput,
        llm_validator=validator,
        error_formatter=_format_errors,
        config=RetryConfig(max_retries=1),
    )

    _output, tokens, _errors = asyncio.run(loop.run(task="test", context="context"))

    assert tokens == 7
