"""The one place "N audited rounds" becomes a raw AgentLoop iteration budget.

`AgentLoop` spends its retry budget once per NODE EXECUTION, not once per pass
through the graph (see `loop.py`'s `while budget.try_consume()`). So a caller
thinking in rounds -- "let the auditor's findings go back to the producer three
times" -- must multiply by the node count before handing the number to
`LoopConfig.max_iter`.

Getting this wrong is a documented, repeated defect in this project rather than a
hypothetical: Stage 3's Phase 1 loop once ran with `max_iter=3` against a 3-node
graph, which afforded exactly one pass and made every retry edge unreachable --
validation errors and audit findings were computed, logged, and discarded. Stage 2
had the same class of bug with a raw 5, which bought one full pass plus two nodes
of a second, so the producer received the auditor's fixes once and nothing ever
re-checked the corrected model.

The conversion lived in three separate places (a literal in Stage 1, plus a
near-identical helper in each of Stage 2 and Stage 3), which is what made the
same mistake reachable three times. It lives here now; the stage-local helpers
delegate and keep only their own node count and their own rationale.

Note this is an UPPER BOUND on rounds, not a promise of them: a pass that exits
early -- a hard deterministic rejection routing back to the producer without
reaching the auditor -- costs fewer than `node_count` units, so the same budget
buys more short passes than full ones.
"""

from __future__ import annotations


def rounds_to_max_iter(rounds: int, node_count: int) -> int:
    """Convert "N full rounds through a `node_count`-node graph" into the raw
    per-node-execution budget `AgentLoop` actually counts.

    `rounds=1` means one pass and no retry. Both arguments are floored at 1, so a
    caller passing 0 or a negative round count still gets a loop that runs rather
    than one that returns `final_output=None` without executing anything.
    """
    return max(1, rounds) * max(1, node_count)
