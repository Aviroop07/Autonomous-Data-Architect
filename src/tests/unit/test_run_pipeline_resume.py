"""run_pipeline can reuse a previous run's stage artifacts.

Stage artifacts were always WRITTEN and never read back, so `--stages 3` still
re-ran Stage 1 from scratch, and a Stage 3 crash re-paid the entire Stage 1+2
LLM cost -- by far the most expensive part of a run -- to regenerate output that
was already sitting on disk.

--resume-from supplies the stages that --stages does not recompute. These tests
make no LLM calls: they assert the artifact plumbing and the error messages,
which is where a resume feature actually goes wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_pipeline
from src.orchestration.stage1.models import Output as Stage1Output
from src.orchestration.stage2.models import Output as Stage2Output


def _empty_plan():
    """A minimal but VALID chunk plan, built from the real model."""
    from src.pipeline.stage2.models.chunk import ChunkedPlan

    return ChunkedPlan(chunks=[[]], core_modeling_facts=[])


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestLoadStageArtifact:
    def test_missing_file_raises_a_message_naming_the_fix(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            run_pipeline._load_stage_artifact(
                tmp_path, "stage1.json", Stage1Output, "Stage 1"
            )
        message = str(exc.value)
        assert "stage1.json" in message
        assert "--stages" in message, "the error must say how to recover"

    def test_round_trips_through_the_real_model(self, tmp_path):
        """Dump a genuine Stage1Output and read it back, rather than
        hand-writing a payload. A hand-written dict silently drifts from the
        model -- my first attempt at this test omitted two required fields and
        failed for that reason, not because resume was broken."""
        from src.pipeline.stage1.models.atomic_fact import AtomicFact

        original = Stage1Output(
            original_nl="a specification",
            domain="Test Domain",
            analytical_goal="a goal",
            final_facts=[
                AtomicFact(id=1, fact="An entity has an attribute.", tags=[])
            ],
            iterations=[],
            plan=_empty_plan(),
        )
        _write(tmp_path / "stage1.json", original.model_dump(mode="json"))
        loaded = run_pipeline._load_stage_artifact(
            tmp_path, "stage1.json", Stage1Output, "Stage 1"
        )
        assert loaded.domain == "Test Domain"
        assert loaded.analytical_goal == "a goal"
        assert [f.id for f in loaded.final_facts] == [1]


def _newest_complete_run() -> Path | None:
    for candidate in reversed(sorted(Path("artifacts/runs").glob("pipeline_*"))):
        if (candidate / "stage1.json").exists() and (candidate / "stage2.json").exists():
            return candidate
    return None


class TestResumeFromARealCompletedRun:
    """Against genuine artifacts from a completed live run, when one exists."""

    def test_both_stages_load_and_carry_their_content(self):
        run = _newest_complete_run()
        if run is None:
            pytest.skip("no completed run with stage1.json + stage2.json on disk")
        s1 = run_pipeline._load_stage_artifact(
            run, "stage1.json", Stage1Output, "Stage 1"
        )
        s2 = run_pipeline._load_stage_artifact(
            run, "stage2.json", Stage2Output, "Stage 2"
        )
        assert s1.final_facts, "resumed Stage 1 should carry its facts"
        assert s2.final_global_schema is not None
        assert s2.final_global_schema.tables, "resumed schema should have tables"

    def test_resumed_schema_is_usable_as_stage3_input(self):
        """The point of resuming: Stage 3 can run on it without Stage 1 or 2."""
        run = _newest_complete_run()
        if run is None:
            pytest.skip("no completed run on disk")
        s2 = run_pipeline._load_stage_artifact(
            run, "stage2.json", Stage2Output, "Stage 2"
        )
        schema = s2.final_global_schema
        assert schema is not None
        errors = schema._validate()
        assert not errors, f"a resumed schema must still be valid: {errors[:3]}"


class TestRefusesToSkipWithoutASource:
    @pytest.mark.asyncio
    async def test_excluding_stage1_without_resume_from_is_rejected(self, tmp_path):
        import argparse

        args = argparse.Namespace(
            input="hospital",
            stages="2,3",
            resume_from=None,
            out=str(tmp_path / "out"),
            model=None,
            dump_artifacts=False,
        )
        with pytest.raises(SystemExit) as exc:
            await run_pipeline.run(args)
        assert "--resume-from" in str(exc.value)
