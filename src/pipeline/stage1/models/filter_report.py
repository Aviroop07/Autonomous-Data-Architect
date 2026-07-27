from typing import List

from pydantic import Field

from src.pipeline.stage1.models.raw_fact import RawFact
from src.util.orchestration.loop_types import LoopOutputModel


class FilterReport(LoopOutputModel):
    """Deterministic structural-check verdict for one enrichment round.

    Emitted by the context-filter loop node. `get_errors()` surfaces only the
    invalid-reference messages -- those are the sole structural problem the
    enricher must act on (self-references are auto-repaired and duplicates are
    auto-dropped, so neither counts as an error that drives a retry).
    """

    should_exit: bool = Field(
        description=(
            "True if the enrichment loop should terminate: no invalid-reference "
            "facts remain AND the last auditor verdict was acceptable."
        )
    )
    accepted_facts: List[RawFact] = Field(
        default_factory=list,
        description=(
            "Proposed external facts that passed all structural checks this round "
            "(self-references stripped, duplicates removed)."
        ),
    )
    structural_errors: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable messages for invalid-reference facts the enricher "
            "must re-anchor or drop on the next round."
        ),
    )
    self_ref_repaired: int = Field(
        default=0,
        description="Count of facts whose self-reference was deterministically stripped.",
    )
    duplicates_dropped: int = Field(
        default=0,
        description="Count of duplicate external facts dropped this round.",
    )
    invalid_reference_count: int = Field(
        default=0,
        description="Count of facts referencing no valid original (non-external) fact.",
    )

    def get_errors(self) -> list[str]:
        return list(self.structural_errors)
