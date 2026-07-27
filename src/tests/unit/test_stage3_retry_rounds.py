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
    """The default. One initial pass plus two retries."""
    assert rounds_to_max_iter(3) == 3 * GENERATOR_GRAPH_NODE_COUNT


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
    # Default is 3 rounds; anything > one pass proves a retry is reachable.
    assert rounds_to_max_iter(3) > GENERATOR_GRAPH_NODE_COUNT
