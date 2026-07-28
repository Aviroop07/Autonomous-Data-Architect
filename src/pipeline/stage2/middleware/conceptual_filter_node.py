from typing import List, Optional, Tuple
from src.util.orchestration.loop_types import (
    LoopAgent,
    LoopContext,
    LoopOutputModel,
    HistoryEntry,
)
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel


class ConceptualFilterReport(LoopOutputModel):
    """Verdict of the deterministic structural filter.

    `det_errors` are HARD errors: the model is structurally broken and must be
    regenerated. `advisories` are soft observations that may well be correct as
    they stand, so they must never fail the model -- they are handed to the
    generator as context on its next natural turn instead.

    Collapsing the two is what made the shard loop unpredictable. The FK-name
    heuristic below is a guess by construction, and it used to set
    is_valid=False, so a legitimately-named attribute (an order number, a
    product code) bounced the model straight back to the extractor. In a live
    retail run that consumed the first two of five rounds before the auditor
    had been reached even once -- the heuristic re-fires up to `threshold`
    times per attribute, so the waste was exactly as large as that constant.
    """

    is_valid: bool = True
    det_errors: List[str] = []
    advisories: List[str] = []

    def get_errors(self) -> list[str]:
        # Only hard errors -- an advisory is not an unresolved issue, and
        # counting it as one made the loop's own retry accounting wrong too.
        return self.det_errors


class ConceptualFilterLoopAgent(LoopAgent):
    """
    Deterministic structural filter for the ER generation loop.
    Checks the ConceptualModel for hard structural errors and tracks soft warnings
    for suspiciously named foreign key attributes.
    """

    def __init__(self, fact_ids: Optional[List[int]] = None) -> None:
        self._warning_counts: dict[Tuple[str, str], int] = {}
        self._last_model: Optional[ConceptualModel] = None
        self._fact_ids: List[int] = list(fact_ids or [])

    async def invoke(self, query: str) -> Tuple[LoopOutputModel, int]:
        report = ConceptualFilterReport()
        model = self._last_model
        if model is None:
            report.is_valid = False
            report.det_errors.append("Model is missing or invalid.")
            return report, 0

        # Hard structural errors
        errors = model.get_errors()
        if errors:
            report.is_valid = False
            report.det_errors.extend(errors)

        suspicious_attributes = self._check_suspicious_attributes(model)

        if suspicious_attributes:
            report.advisories.append(
                "These attributes match a foreign-key naming heuristic:\n"
                + "\n".join([f"- {attr}" for attr in suspicious_attributes])
                + "\n\nThe mapper synthesizes every foreign key from the relationships, so "
                "an attribute that exists only to point at another entity should be a "
                "relationship instead. This is a name-based guess and may be wrong: a "
                "domain-meaningful identifier that the source material describes as the "
                "entity's own property is correct as it stands. Remove the ones that are "
                "references and keep the rest."
            )

        unreferenced = self._unreferenced_fact_ids(model)
        if unreferenced:
            report.advisories.append(
                "No entity, attribute or relationship cites these fact IDs in its "
                "source_fact_ids: "
                + ", ".join(str(i) for i in unreferenced)
                + ". Either the fact is not represented in the model at all -- in which "
                "case model it -- or it is represented but its provenance was not "
                "recorded, in which case add the ID to every element it describes. A "
                "purely statistical or distributional fact that constrains no structure "
                "may legitimately have no home here; leave those alone."
            )

        return report, 0

    def _unreferenced_fact_ids(self, model: ConceptualModel) -> List[int]:
        """Fact IDs the model cites nowhere.

        This replaces an auditor instruction that asked an LLM to verify a
        many-to-many fact/element mapping by reading it, which in a live run
        produced the unsatisfiable "add source_fact_ids to all attributes where
        they are currently an empty list" -- a restatement of the symptom with no
        fact ID in it. Set arithmetic answers the answerable part of that question
        exactly, and names the IDs.

        Deliberately an advisory rather than a hard error: a shard's facts include
        distributional statements that constrain no ER structure, so an
        unreferenced ID is not proof of a defect and must not cost a retry round.
        """
        if not self._fact_ids:
            return []
        cited: set[int] = set()
        for entity in model.entities:
            cited.update(entity.source_fact_ids or [])
            for attribute in entity.attributes:
                cited.update(attribute.source_fact_ids or [])
        for rel in model.relationships:
            cited.update(rel.source_fact_ids or [])
            for attribute in rel.attributes or []:
                cited.update(attribute.source_fact_ids or [])
        return [i for i in self._fact_ids if i not in cited]

    def build_context(self, ctx: LoopContext) -> str:
        produced = ctx.node_outputs.get("extractor")
        self._last_model = produced if isinstance(produced, ConceptualModel) else None
        return ""

    def _check_suspicious_attributes(self, model: ConceptualModel) -> List[str]:
        threshold = 2
        suspicious_attributes: List[str] = []

        for e in model.entities:
            for a in e.attributes:
                a_low = a.name.lower()
                # Exclude primary keys
                if a_low in [pk.lower() for pk in e.identifier_attributes]:
                    continue

                is_suspicious = False
                # Heuristic 1: Common FK suffixes and prefixes
                if any(
                    a_low.endswith(s)
                    for s in ["_id", "_number", "_code", "_ref", "_key"]
                ):
                    is_suspicious = True
                elif any(a_low.startswith(p) for p in ["fk_", "ref_"]):
                    is_suspicious = True
                else:
                    # Heuristic 2: Contains another entity's name (case insensitive)
                    # Heuristic 3: Matches another entity's PK name exactly
                    for other_e in model.entities:
                        if other_e.name.lower() == e.name.lower():
                            continue

                        if other_e.name.lower() in a_low:
                            is_suspicious = True
                            break

                        if a_low in [
                            pk.lower() for pk in other_e.identifier_attributes
                        ]:
                            is_suspicious = True
                            break

                if is_suspicious:
                    key = (e.name, a.name)
                    self._warning_counts[key] = self._warning_counts.get(key, 0) + 1

                    if self._warning_counts[key] <= threshold:
                        suspicious_attributes.append(f"Entity '{e.name}': '{a.name}'")

        return suspicious_attributes

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: LoopOutputModel | None,
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, ConceptualFilterReport)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=(
                f"Valid ({len(output.advisories)} advisory)"
                if output.is_valid and output.advisories
                else "Valid"
                if output.is_valid
                else f"Found {len(output.det_errors)} structural issues"
            ),
        )
