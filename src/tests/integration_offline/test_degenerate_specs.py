"""Stage 1 on the two smallest possible inputs.

Both run the real `orchestrate()` -- the real AgentLoop, the real extraction
validator, the real tagger conversion, the real chunker. The assertions state
what Stage 1 actually returns, rather than merely that it did not raise.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

from src.orchestration.stage1.models import Output as Stage1Output
from src.pipeline.stage1.models.coverage_report import CoverageReport
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput, TaggerOutput
from src.tests.fixtures.canned_llm import CannedAgentProvider
from src.tests.fixtures.canned_payloads import stage1 as p1

from .conftest import pin_context_window


def _run(spec: str, provider: CannedAgentProvider) -> Tuple[Stage1Output, int]:
    from src.orchestration.stage1.entry import orchestrate as stage1

    pin_context_window()
    return asyncio.run(stage1(spec, provider=provider))


def test_an_empty_spec_yields_an_empty_but_well_formed_output() -> None:
    """An empty specification must produce a complete, empty `Output` -- not an
    exception, and not a half-built object.

    Stage 1's degenerate path bails out early before any LLM call: calling the
    tagger or chunker on an empty fact list would waste budget and the resulting
    empty chunk gives Stage 2 a hallucination opportunity. The plan is still
    structurally usable by Stage 2 rather than absent.
    """
    provider = (
        CannedAgentProvider()
        .script(RephrasedOutput, p1.empty_extraction)
        .script(IntegrityReport, p1.clean_integrity_report)
        .script(CoverageReport, p1.complete_coverage_report)
        .script(TaggerOutput, p1.empty_tagger_output)
    )
    output, tokens = _run("", provider)

    assert output.final_facts == []
    assert output.original_nl == ""
    assert output.domain == "Unknown"
    assert output.analytical_goal == "General Purpose"
    assert output.converged is False
    # The plan must still be structurally usable by Stage 2 rather than absent.
    assert output.plan.core_modeling_facts == []
    assert output.plan.chunks == []
    # Zero-fact early return: no LLM calls were made.
    assert provider.calls == [], provider.calls
    assert tokens == 0


def test_a_single_fact_spec_produces_exactly_one_tagged_fact_in_one_chunk() -> None:
    """One sentence in, one fully-formed `AtomicFact` out, in one chunk.

    The single-fact case is the boundary where the chunker's packing loop, the
    tagger's id matching, and the segment-offset carry all have exactly one item
    to work with -- so an off-by-one in any of them is invisible on a six-fact
    input but fatal here.

    Mutation this catches: have `convert_to_atomic` match tags by identity rather
    than `str(t.id) == str(raw.id)`, or drop the `segment_lookup` argument at the
    `orchestrate()` call site -- the fact still arrives, but with the default
    STRUCTURAL fallback tag and `start_char == -1`, losing the span that Stage 2's
    provenance and the evaluation's span metrics both depend on.
    """
    provider = (
        CannedAgentProvider()
        .script(RephrasedOutput, p1.single_fact_extraction)
        .script(IntegrityReport, p1.clean_integrity_report)
        .script(CoverageReport, p1.complete_coverage_report)
        .script(TaggerOutput, p1.single_fact_tagger_output)
    )
    output, _tokens = _run(p1.SINGLE_FACT_SPEC, provider)

    assert len(output.final_facts) == 1, (
        f"expected exactly one fact, got {[f.id for f in output.final_facts]}"
    )
    fact = output.final_facts[0]
    assert fact.id == 1
    assert fact.fact == "A customer has a name."
    assert [t.value for t in fact.tags] == ["STRUCTURAL"]
    # The segment span must have survived, not fallen back to the (-1, -1) marker.
    assert fact.segment_text == p1.SINGLE_FACT_SPEC
    assert (fact.start_char, fact.end_char) == (0, len(p1.SINGLE_FACT_SPEC))

    assert len(output.plan.chunks) == 1, (
        f"one fact must yield exactly one chunk, got {len(output.plan.chunks)}"
    )
    assert [f.id for f in output.plan.chunks[0]] == [1]
