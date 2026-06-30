from pathlib import Path
from typing import List, Optional

from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.orchestration.loop_types import (
    HistoryEntry,
    LoopAgent,
    LoopContext,
    LoopOutputModel,
)
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType, Severity
from src.pipeline.stage1.middleware.error_formatter import format_errors_for_stage1
from src.pipeline.stage1.middleware.validation import deterministic_validator
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput
from src.pipeline.stage1.models.integrity_report import IntegrityReport

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def build_extractor_agent(
    system_prompt: str,
    model: Optional[str] = None,
) -> AgentType:
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=RephrasedOutput,
        model=model,
        name="atomic_fact_extractor",
    )


class FactExtractorLoopAgent(LoopAgent):
    """LoopAgent for the atomic fact extractor node.

    Absorbs the old _ExtractorContextBuilder stateful logic. Runs
    deterministic_validator on the prior output inside build_context so
    NL-scoped validation works correctly without extra infrastructure.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model
        self._agent: Optional[AgentType] = None
        self._last_det_errors: List[ErrorRecord] = []
        self._errored_ids_history: set[int] = set()

    def _get_agent(self) -> AgentType:
        if self._agent is None:
            system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
            self._agent = build_extractor_agent(
                system_prompt=system_prompt, model=self._model
            )
        return self._agent

    def _compute_segment_offsets(self, parsed: RephrasedOutput, text: str) -> None:
        from src.util.algorithms.semantic_match import FactOriginMatcher

        cursor = 0
        normalized_text = text.lower()
        matcher = None

        for segment in parsed.segments:
            if not segment.text:
                continue

            seg_lower = segment.text.lower()
            idx = normalized_text.find(seg_lower, cursor)

            if idx != -1:
                # Exact match found
                segment.start_char = idx
                segment.end_char = idx + len(segment.text)
                cursor = segment.end_char
            else:
                # Fallback to fuzzy matcher
                if matcher is None:
                    matcher = FactOriginMatcher(text)
                best_span_res = matcher.find_best_source_span(segment.text)
                if best_span_res:
                    best_span = best_span_res.span
                    if best_span.char_start >= cursor:
                        segment.start_char = best_span.char_start
                        segment.end_char = best_span.char_end
                        cursor = best_span.char_end
                    else:
                        # found a match but it was before cursor (overlap or out of order)
                        segment.start_char = best_span.char_start
                        segment.end_char = best_span.char_end

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        parsed, tokens = await get_response(
            agent=self._get_agent(),
            output_structure=RephrasedOutput,
            query=query,
        )
        assert isinstance(parsed, RephrasedOutput)
        if hasattr(self, "_last_nl_description") and self._last_nl_description:
            self._compute_segment_offsets(parsed, self._last_nl_description)
        return parsed, tokens

    def build_context(self, ctx: LoopContext) -> str:
        self._last_nl_description = ctx.initial_context
        nl_description = ctx.initial_context

        extractor_output: Optional[RephrasedOutput] = None
        raw_extractor = ctx.node_outputs.get("extractor")
        if isinstance(raw_extractor, RephrasedOutput):
            extractor_output = raw_extractor

        verifier_output: Optional[IntegrityReport] = None
        raw_verifier = ctx.node_outputs.get("verifier")
        if isinstance(raw_verifier, IntegrityReport):
            verifier_output = raw_verifier

        # Run deterministic validation on the previous extractor output.
        if extractor_output is not None:
            self._last_det_errors = deterministic_validator(
                extractor_output, nl_description
            )
            for e in self._last_det_errors:
                if e.fact_id is not None:
                    self._errored_ids_history.add(e.fact_id)

        # Accumulate errored IDs from verifier report.
        if verifier_output is not None:
            for issue_list in [
                verifier_output.missing_information,
                verifier_output.introduced_information,
                verifier_output.changed_constraints,
                verifier_output.unresolved_ambiguities,
            ]:
                for issue in issue_list:
                    if issue.fact_id is not None:
                        self._errored_ids_history.add(issue.fact_id)

        # Accepted facts: facts with no errors across all rounds.
        accepted_section: Optional[str] = None
        if extractor_output is not None and self._errored_ids_history:
            accepted = [
                f
                for f in extractor_output.flat_facts
                if f.id not in self._errored_ids_history
            ]
            if accepted:
                lines = [f"- id: {f.id}\n  fact: {f.fact}" for f in accepted]
                accepted_section = "\n".join(lines)

        # Reconstruct ErrorRecord list for the rich Stage 1 error formatter.
        all_errors: List[ErrorRecord] = list(self._last_det_errors)
        if verifier_output is not None:
            for issue in verifier_output.missing_information:
                all_errors.append(
                    ErrorRecord(
                        iteration=ctx.iteration,
                        error_type=ErrorType.MISSING,
                        severity=Severity(issue.severity.value),
                        description=issue.description,
                        fact_id=issue.fact_id,
                    )
                )
            for issue in verifier_output.introduced_information:
                all_errors.append(
                    ErrorRecord(
                        iteration=ctx.iteration,
                        error_type=ErrorType.INTRODUCED,
                        severity=Severity(issue.severity.value),
                        description=issue.description,
                        fact_id=issue.fact_id,
                    )
                )
            for issue in verifier_output.changed_constraints:
                all_errors.append(
                    ErrorRecord(
                        iteration=ctx.iteration,
                        error_type=ErrorType.CHANGED,
                        severity=Severity(issue.severity.value),
                        description=issue.description,
                        fact_id=issue.fact_id,
                    )
                )
            for issue in verifier_output.unresolved_ambiguities:
                all_errors.append(
                    ErrorRecord(
                        iteration=ctx.iteration,
                        error_type=ErrorType.DETERMINISTIC,
                        severity=Severity(issue.severity.value),
                        description=f"[Ambiguity] {issue.description}",
                        fact_id=issue.fact_id,
                        signature_key=f"ambiguity:{issue.fact_id}",
                    )
                )

        parts: List[str] = [f"## NL DESCRIPTION\n{nl_description}"]

        if accepted_section is not None:
            parts.append(
                f"## ACCEPTED FACTS (keep these unchanged)\n{accepted_section}\n"
                "These facts have passed all previous validation checks. "
                "Do NOT regenerate, reword, or remove them. Include them verbatim in your output."
            )

        if all_errors:
            parts.append(
                format_errors_for_stage1(
                    all_errors,
                    ctx.iteration,
                    extractor_output or RephrasedOutput(segments=[]),
                    nl_description,
                )
            )

        parts.append("## TASK\nExtract atomic facts from the description")
        return "\n\n".join(parts)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, RephrasedOutput)
        count = len(output.flat_facts)
        if prior is None:
            return HistoryEntry(
                round=round_num,
                node=node,
                changes_summary=f"extracted {count} facts",
                was_improvement=None,
            )
        assert isinstance(prior, RephrasedOutput)
        delta = count - len(prior.flat_facts)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=f"{count} facts ({delta:+d} vs prior)",
            was_improvement=(delta != 0),
        )
