"""Stage 3's Phase 1 loop must actually be able to retry.

AgentLoop's budget is consumed once per NODE EXECUTION. The generator graph has
three nodes, and Phase 1 previously passed a raw max_iter of 3 -- exactly one
pass, so the det_checker->generator and auditor->generator retry edges could
never fire. Validation errors and audit findings were computed, written to
history, and then discarded.
"""

from __future__ import annotations

from src.orchestration.stage3.extraction import (
    GENERATOR_GRAPH_NODE_COUNT,
    _build_generator_loop_config,
    rounds_to_max_iter,
)


def test_node_count_constant_matches_the_real_graph():
    """Guards the conversion against someone adding a node to the graph."""
    config = _build_generator_loop_config(max_iter=1, model=None)
    assert len(config.agents) == GENERATOR_GRAPH_NODE_COUNT


def test_one_round_is_one_pass_through_every_node():
    assert rounds_to_max_iter(1) == GENERATOR_GRAPH_NODE_COUNT


def test_three_rounds_affords_two_genuine_retries():
    assert rounds_to_max_iter(3) == 3 * GENERATOR_GRAPH_NODE_COUNT


def test_two_rounds_affords_one_genuine_retry():
    """The default, as of the 2-vs-3 cost/recall tradeoff: one initial pass
    plus one retry. Measured on the hospital spec (experiments/
    stage3_recall_rate.py), 3 rounds took fanout recall 92% -> 100% but cost
    ~2.5x the tokens/time of the old always-one-pass behaviour; recall was
    already saturated by the second round in every run that needed a retry at
    all. 2 rounds is a considered default trading some of that headroom back
    for ~1.7x instead of ~2.5x -- not a re-measured one, so revisit with a
    direct 2-vs-3 comparison before an eval campaign if the budget allows."""
    assert rounds_to_max_iter(2) == 2 * GENERATOR_GRAPH_NODE_COUNT


def test_rounds_below_one_still_yields_a_usable_budget():
    """A caller passing 0 or a negative should not produce a loop that cannot
    execute a single node."""
    assert rounds_to_max_iter(0) >= GENERATOR_GRAPH_NODE_COUNT
    assert rounds_to_max_iter(-5) >= GENERATOR_GRAPH_NODE_COUNT


def test_default_phase1_budget_permits_a_retry_edge_to_fire():
    """The regression proper: the default must exceed one full pass, otherwise
    a det_checker error can never route back to the generator."""
    import inspect

    from src.orchestration.stage3 import entry

    source = inspect.getsource(entry.orchestrate)
    assert "rounds_to_max_iter" in source, (
        "Phase 1 must convert rounds to raw iterations through the shared "
        "helper, not with an inline multiplier"
    )
    assert entry._DEFAULT_PHASE1_ROUNDS >= 2, (
        "the default must afford at least one retry beyond the first pass"
    )
    assert rounds_to_max_iter(entry._DEFAULT_PHASE1_ROUNDS) > GENERATOR_GRAPH_NODE_COUNT
