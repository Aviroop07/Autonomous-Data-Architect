from typing import List

from pydantic import Field

from src.util.orchestration.loop_types import LoopOutputModel


class ExtractionValidationReport(LoopOutputModel):
    """Deterministic structural-check verdict for one extraction round."""

    is_clean: bool = Field(
        description=(
            "True if the extracted facts passed all structural checks (e.g., verbatim origin texts)."
        )
    )
    structural_errors: List[str] = Field(
        default_factory=list,
        description=(
            "Human-readable messages for structural errors the extractor must fix."
        ),
    )

    def get_errors(self) -> list[str]:
        return list(self.structural_errors)
