"""Deterministic offline unit tests for Stage 4 (code generation) models.

Targets: src/pipeline/stage4/models.py
Covers TableParameters (incl. the positive-scale field_validators),
ParameterManifest, and SynthesisResult (defaults + round-trip).

No LLM / no network. Source is NOT modified.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline.stage4.models import (
    TableParameters,
    ParameterManifest,
    SynthesisResult,
)


# --------------------------------------------------------------------------- #
# TableParameters - valid construction
# --------------------------------------------------------------------------- #

def test_table_parameters_anchor_with_n_seeds():
    p = TableParameters(table_name="USER", n_seeds=100)
    assert p.table_name == "USER"
    assert p.n_seeds == 100
    assert p.avg_fanout is None
    # sparsity defaults to an empty dict (not None)
    assert p.sparsity == {}


def test_table_parameters_dependent_with_fanout():
    p = TableParameters(table_name="ORDER_ITEM", avg_fanout=3.5)
    assert p.avg_fanout == pytest.approx(3.5)
    assert p.n_seeds is None


def test_table_parameters_both_optional_scales_may_be_none():
    # Both scale params are Optional and default to None.
    p = TableParameters(table_name="ISOLATED")
    assert p.n_seeds is None
    assert p.avg_fanout is None


def test_table_parameters_sparsity_roundtrip():
    p = TableParameters(
        table_name="LOAN",
        n_seeds=50,
        sparsity={"comment": 0.3, "second_signer": 0.9},
    )
    assert p.sparsity["comment"] == pytest.approx(0.3)
    assert p.sparsity["second_signer"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# TableParameters - validator rejects non-positive scales
# --------------------------------------------------------------------------- #

def test_table_parameters_zero_n_seeds_rejected():
    with pytest.raises(ValidationError):
        TableParameters(table_name="USER", n_seeds=0)


def test_table_parameters_negative_n_seeds_rejected():
    with pytest.raises(ValidationError):
        TableParameters(table_name="USER", n_seeds=-5)


def test_table_parameters_zero_avg_fanout_rejected():
    with pytest.raises(ValidationError):
        TableParameters(table_name="ORDER", avg_fanout=0.0)


def test_table_parameters_negative_avg_fanout_rejected():
    with pytest.raises(ValidationError):
        TableParameters(table_name="ORDER", avg_fanout=-2.0)


def test_table_parameters_validation_error_mentions_positive():
    # The validator message should surface the "positive" rule.
    with pytest.raises(ValidationError) as exc:
        TableParameters(table_name="USER", n_seeds=-1)
    assert "positive" in str(exc.value).lower()


def test_table_parameters_small_positive_scales_accepted():
    # Boundary: strictly positive values must pass.
    p = TableParameters(table_name="X", n_seeds=1, avg_fanout=0.001)
    assert p.n_seeds == 1
    assert p.avg_fanout == pytest.approx(0.001)


# --------------------------------------------------------------------------- #
# ParameterManifest
# --------------------------------------------------------------------------- #

def test_parameter_manifest_holds_parameters_and_reasoning():
    manifest = ParameterManifest(
        parameters=[
            TableParameters(table_name="USER", n_seeds=100),
            TableParameters(table_name="ORDER", avg_fanout=2.0),
        ],
        reasoning="Users are anchors; orders fan out per user.",
    )
    assert len(manifest.parameters) == 2
    assert manifest.parameters[0].table_name == "USER"
    assert "anchors" in manifest.reasoning


def test_parameter_manifest_requires_reasoning():
    with pytest.raises(ValidationError):
        ParameterManifest(parameters=[])  # type: ignore[call-arg]


def test_parameter_manifest_empty_parameter_list_allowed():
    manifest = ParameterManifest(parameters=[], reasoning="No tables.")
    assert manifest.parameters == []


# --------------------------------------------------------------------------- #
# SynthesisResult - defaults and round-trip
# --------------------------------------------------------------------------- #

def test_synthesis_result_defaults():
    r = SynthesisResult(generated_code="print('hi')")
    assert r.generated_code == "print('hi')"
    assert r.token_usage == 0
    assert r.success is True
    assert r.error_message is None
    assert r.verification_status is None
    assert r.verification_logs == []
    assert r.column_coverage == pytest.approx(0.0)


def test_synthesis_result_failure_fields():
    r = SynthesisResult(
        generated_code="",
        token_usage=1234,
        success=False,
        error_message="boom",
        verification_status="FAILED",
        verification_logs=["step 1 failed", "rolled back"],
        column_coverage=0.42,
    )
    assert r.success is False
    assert r.error_message == "boom"
    assert r.verification_status == "FAILED"
    assert r.verification_logs == ["step 1 failed", "rolled back"]
    assert r.column_coverage == pytest.approx(0.42)


def test_synthesis_result_roundtrip_model_dump():
    r = SynthesisResult(
        generated_code="x = 1",
        token_usage=10,
        verification_status="PASSED",
        column_coverage=1.0,
    )
    dumped = r.model_dump()
    rebuilt = SynthesisResult(**dumped)
    assert rebuilt == r
    assert dumped["verification_status"] == "PASSED"
    assert dumped["column_coverage"] == pytest.approx(1.0)
