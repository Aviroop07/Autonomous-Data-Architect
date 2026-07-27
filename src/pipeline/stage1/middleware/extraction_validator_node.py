"""Deterministic structural-check node for the fact-extraction loop."""

from typing import Optional

from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.pipeline.stage1.middleware.validation import check_verbatim_substring
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput
from src.pipeline.stage1.models.extraction_validation_report import (
    ExtractionValidationReport,
)


class ExtractionValidatorLoopAgent(LoopAgent):
    """Deterministic structural-check node for the fact-extraction loop.

    Wraps the verbatim substring validation as a `LoopAgent`. This ensures the
    extraction loop can validate proposed segments each round and route
    structural problems back to the extractor constructively, rather than only
    advising the verifier.

    Policy per round:
    - check_verbatim_substring verifies all segments.
    - If valid, `is_clean` is True.
    - Otherwise, `is_clean` is False and `structural_errors` are surfaced.
    """

    def __init__(self) -> None:
        self._proposed: Optional[RephrasedOutput] = None
        self._initial_context = ""

    def build_context(self, ctx: LoopContext) -> str:
        extractor_output = ctx.node_outputs.get("extractor")
        if isinstance(extractor_output, RephrasedOutput):
            self._proposed = extractor_output
        else:
            self._proposed = None
        self._initial_context = ctx.initial_context
        return "## DETERMINISTIC EXTRACTION CHECK"

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        if not self._proposed:
            return ExtractionValidationReport(is_clean=True, structural_errors=[]), 0

        errors = check_verbatim_substring(self._proposed.segments, self._initial_context)
        
        is_clean = len(errors) == 0
        structural_errors = [e.description for e in errors]

        report = ExtractionValidationReport(
            is_clean=is_clean,
            structural_errors=structural_errors,
        )
        return report, 0

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, ExtractionValidationReport)
        summary = (
            f"Validation clean: {output.is_clean}, "
            f"Errors: {len(output.structural_errors)}"
        )
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=summary,
            was_improvement=output.is_clean,
        )
