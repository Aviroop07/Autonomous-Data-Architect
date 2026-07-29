"""A misextraction fix must not destroy constraints nobody complained about.

Found on the first live Stage 3 run to complete (2026-07-29). The shard extracted
a well-formed fanout constraint on ORDER -> ORDER_ITEM from fact 32. Three
UNRELATED misextraction fixes then triggered one shard re-run, and the final
output contained ZERO structural constraints. The lost constraint appeared in no
conflict, no dismissal, and no unsupported list -- it was simply gone.

The cause was scope. A fix is addressed to specific FACTS (the reconciler names
them, "fact 32: ..."), but the repair re-runs the whole shard and the result
REPLACED the entire shard output. Extraction is not deterministic, so a re-run
routinely fails to reproduce constraints it got right the first time.

The rule these tests pin: the re-run wins for any constraint touching a fact it
was asked to fix, and for anything it independently re-emitted; a prior
constraint citing NONE of the fixed facts that the re-run did not produce is
carried forward, because nothing ever authorised dropping it.
"""

from __future__ import annotations

from typing import Literal, Sequence

from src.orchestration.stage3.reconciliation import _merge_rerun_output
from src.pipeline.stage3.models.cross_shard import Constraint, UnifiedExtractionOutput
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.relation.nodes import BaseTable


def _con(
    facts: Sequence[int],
    category: Literal["statistical", "structural", "logic", "temporal", "derived"] = (
        "structural"
    ),
) -> Constraint:
    return Constraint(
        fact_references=list(facts),
        on=BaseTable(name="ORDER"),
        condition=RComparison(
            op=">", left=RColumnRef(name="total"), right=RLiteral(value=0)
        ),
        category=category,
    )


def _refs(items: Sequence[Constraint]) -> list[list[int]]:
    return [list(i.fact_references) for i in items]


def test_the_measured_loss_no_longer_happens() -> None:
    """The exact shape of the live failure: a structural constraint on fact 32,
    three fixes for other facts, and a re-run that did not re-emit it."""
    prior = UnifiedExtractionOutput(
        structural_constraints=[_con([32])],
        logic_constraints=[_con([59, 60], "logic")],
    )
    rerun = UnifiedExtractionOutput(logic_constraints=[_con([59, 60], "logic")])

    merged = _merge_rerun_output(
        prior=prior, rerun=rerun, fixed_fact_ids={59, 60, 73}, shard_idx=0
    )

    assert _refs(merged.structural_constraints) == [[32]], (
        "a constraint nobody objected to must survive a re-run aimed elsewhere"
    )


def test_an_in_scope_drop_is_honoured_not_resurrected() -> None:
    """The other half, and the reason this cannot just union everything: when the
    re-run WAS asked to reconsider a fact, deleting its constraint is a decision,
    not a loss. Carrying it back would defeat the fix that was requested."""
    prior = UnifiedExtractionOutput(
        logic_constraints=[_con([59, 60], "logic"), _con([32], "logic")]
    )
    rerun = UnifiedExtractionOutput(logic_constraints=[])

    merged = _merge_rerun_output(
        prior=prior, rerun=rerun, fixed_fact_ids={59, 60}, shard_idx=0
    )

    assert _refs(merged.logic_constraints) == [[32]]


def test_the_rerun_version_supersedes_a_restatement() -> None:
    """Same facts in the same category is the same rule restated -- exactly the
    case where the repaired version must win, not be duplicated alongside."""
    prior = UnifiedExtractionOutput(structural_constraints=[_con([32])])
    fixed = _con([32])
    fixed.severity = "soft"
    rerun = UnifiedExtractionOutput(structural_constraints=[fixed])

    merged = _merge_rerun_output(
        prior=prior, rerun=rerun, fixed_fact_ids={32}, shard_idx=0
    )

    assert len(merged.structural_constraints) == 1, "must not duplicate the rule"
    assert merged.structural_constraints[0].severity == "soft", "re-run's version wins"


def test_a_restatement_wins_even_when_the_fact_was_not_in_scope() -> None:
    """Identity is per-category fact set, independent of scope: if the re-run
    re-emitted the rule at all, that is the current version."""
    prior = UnifiedExtractionOutput(structural_constraints=[_con([7])])
    restated = _con([7])
    restated.severity = "soft"
    rerun = UnifiedExtractionOutput(structural_constraints=[restated])

    merged = _merge_rerun_output(
        prior=prior, rerun=rerun, fixed_fact_ids={99}, shard_idx=0
    )
    assert len(merged.structural_constraints) == 1
    assert merged.structural_constraints[0].severity == "soft"


def test_no_constraint_family_is_left_out_of_the_merge() -> None:
    """Drift guard. The merge names each of the seven families explicitly, so a
    family added to UnifiedExtractionOutput and forgotten here would silently
    return to being replaced wholesale -- reintroducing the whole bug for that
    one family, which is exactly how it went unnoticed for structural
    constraints. Checked against the source rather than a parallel list, so
    there is no second place to keep in step."""
    import inspect

    source = inspect.getsource(_merge_rerun_output)
    missing = [
        field for field in UnifiedExtractionOutput.model_fields if field not in source
    ]
    assert not missing, (
        f"these families would still be replaced wholesale by a re-run: {missing}"
    )


def test_an_empty_rerun_preserves_everything_out_of_scope() -> None:
    """The degenerate case worth being explicit about: a re-run that returns
    nothing at all must not wipe the shard."""
    prior = UnifiedExtractionOutput(
        structural_constraints=[_con([1]), _con([2])],
        logic_constraints=[_con([3], "logic")],
    )
    merged = _merge_rerun_output(
        prior=prior,
        rerun=UnifiedExtractionOutput(),
        fixed_fact_ids=set(),
        shard_idx=0,
    )
    assert _refs(merged.structural_constraints) == [[1], [2]]
    assert _refs(merged.logic_constraints) == [[3]]
