from __future__ import annotations

import asyncio
from typing import List

from src.orchestration.stage3.entry import _extract_shard_with_retry
from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.pipeline.stage3.agents.constraint_patch_agent.agent import patch_stage3_output
from src.pipeline.stage3.agents.mathematics_verifier.agent import verify_mathematics
from src.pipeline.stage3.middleware.mathematics import collect_deterministic_math_issues
from src.pipeline.stage3.models.sql_models import BinaryOperand, LLMResponse, SQLGroundedConstraint
from src.pipeline.stage3.models.validation import (
    MathematicsValidationReport,
    Stage3Issue,
    Stage3PatchPlan,
)


class FakeAgent:
    def __init__(self, response: object, tokens: int = 17) -> None:
        self.response = response
        self.tokens = tokens
        self.calls: List[object] = []

    async def ainvoke(self, payload: object):
        self.calls.append(payload)
        return {
            "structured_response": self.response,
            "messages": [{"usage_metadata": {"total_tokens": self.tokens}}],
        }


def _loan_schema() -> Schema:
    return Schema(tables=[
        Table(
            name="LOAN",
            pk="loan_id",
            columns=[
                Column(name="loan_id", data_type="INTEGER"),
                Column(name="missed_payment_count", data_type="INTEGER"),
            ],
        )
    ])


def _constraint_response(column_name: str = "missed_payment_count") -> LLMResponse:
    return LLMResponse(logical_constraints=[
        SQLGroundedConstraint(
            state_query=f"SELECT {column_name} FROM LOAN",
            left_operand=column_name,
            operator="GTE",
            right_operand=BinaryOperand(kind="literal", value=0),
            fact_references=[1],
        )
    ])


def test_deterministic_feasibility_issues_flag_invalid_state_constraint():
    issues = collect_deterministic_math_issues(_constraint_response("missing_column"), _loan_schema())
    assert len(issues) == 1
    assert issues[0].code == "INVALID_STATE_CONSTRAINT"
    assert issues[0].target == "logical_constraints[0]"


def test_mathematics_verifier_wrapper_uses_structured_report():
    report = MathematicsValidationReport(
        is_valid=False,
        issues=[Stage3Issue(
            code="INVALID_STATE_CONSTRAINT",
            severity="critical",
            target="logical_constraints[0]",
            message="State query references a missing column.",
        )],
        reasoning="Invalid state query.",
    )
    fake_agent = FakeAgent(report, tokens=23)

    parsed, tokens = asyncio.run(verify_mathematics(
        table_name="LOAN",
        shard_schema_json=_loan_schema().model_dump_json(),
        grounded_facts=["[ID 1] Missed payment counts must be non-negative."],
        extracted_metadata=_constraint_response("missing_column"),
        deterministic_issues=report.issues,
        verifier=fake_agent,
    ))

    assert parsed is report
    assert tokens == 23
    query = fake_agent.calls[0]["messages"][0].content
    assert "### EXTRACTED STAGE 3 METADATA" in query
    assert "### DETERMINISTIC VALIDATION ISSUES" in query


def test_constraint_patch_agent_wrapper_returns_patch_plan():
    patched_response = _constraint_response()
    report = MathematicsValidationReport(
        is_valid=False,
        issues=[Stage3Issue(
            code="INVALID_STATE_CONSTRAINT",
            severity="critical",
            target="logical_constraints[0]",
            message="State query references a missing column.",
        )],
    )
    patch_plan = Stage3PatchPlan(
        patched_response=patched_response,
        addressed_issue_codes=["INVALID_STATE_CONSTRAINT"],
        rationale="Use the real schema column.",
    )
    fake_agent = FakeAgent(patch_plan, tokens=29)

    parsed, tokens = asyncio.run(patch_stage3_output(
        table_name="LOAN",
        shard_schema_json=_loan_schema().model_dump_json(),
        grounded_facts=["[ID 1] Missed payment counts must be non-negative."],
        extracted_metadata=_constraint_response("missing_column"),
        validation_report=report,
        patcher=fake_agent,
    ))

    assert parsed is patch_plan
    assert tokens == 29
    query = fake_agent.calls[0]["messages"][0].content
    assert "### ORIGINAL STAGE 3 METADATA" in query
    assert "### VALIDATION REPORT" in query


def test_extract_shard_stores_state_table_constraints():
    schema = _loan_schema()
    fact = AtomicFact(
        id=1,
        fact="Missed payment counts must be non-negative.",
        origin="Missed payment counts must be non-negative.",
    )
    extractor = FakeAgent(_constraint_response(), tokens=11)

    manifests, raw_rules, tokens, history, fact_ids = asyncio.run(_extract_shard_with_retry(
        shard_idx=0,
        table_names=["LOAN"],
        global_schema=schema,
        registry=TableFactRegistry(),
        all_facts=[fact],
        extractor=extractor,
        model=None,
        allocated_fids=[1],
    ))

    assert fact_ids == [1]
    assert tokens == 11
    assert len(raw_rules) == 1
    assert raw_rules[0].state_query == "SELECT missed_payment_count FROM LOAN"
    loan_manifest = next(manifest for manifest in manifests if manifest.table_name == "LOAN")
    assert len(loan_manifest.state_constraints) == 1
    constraint = loan_manifest.state_constraints[0]
    assert constraint.left_operand == "missed_payment_count"
    assert constraint.operator == "GTE"
    assert history[-1].error is None
