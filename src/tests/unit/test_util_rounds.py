"""Pins the rounds -> max_iter conversion, a repeated defect in this project.

AgentLoop spends its retry budget once per NODE EXECUTION, not once per pass, so
a caller thinking in "audited rounds" must multiply by the node count. Getting it
wrong has twice shipped a loop that could not retry: Stage 3's Phase 1 ran with
max_iter=3 against a 3-node graph (one pass, every retry edge unreachable), and
Stage 2 ran with a raw 5 (one full pass plus two nodes, so nothing ever
re-checked the corrected model).

The conversion used to exist in three places -- a bare literal in Stage 1 plus a
near-identical helper in each of Stage 2 and Stage 3 -- which is what made the
same mistake reachable three times over. These tests cover the shared helper and
assert each stage's wrapper agrees with it, so a future divergence fails here.
"""

from __future__ import annotations

import pytest

from src.util.orchestration.rounds import rounds_to_max_iter


class TestRoundsToMaxIter:
    def test_one_round_is_exactly_one_pass(self):
        """rounds=1 must buy one pass and NO retry -- not one node."""
        assert rounds_to_max_iter(1, 3) == 3

    @pytest.mark.parametrize(
        "rounds,nodes,expected",
        [
            (3, 3, 9),  # both live Stage 1 loops, and Stage 2/3 at rounds=3
            (2, 3, 6),
            (5, 3, 15),
            (3, 2, 6),  # a 2-node graph
            (4, 5, 20),
        ],
    )
    def test_is_rounds_times_nodes(self, rounds, nodes, expected):
        assert rounds_to_max_iter(rounds, nodes) == expected

    @pytest.mark.parametrize("rounds", [0, -1, -100])
    def test_non_positive_rounds_still_affords_one_pass(self, rounds):
        """A degenerate round count must not yield max_iter=0.

        LoopConfig accepts max_iter=0 and the loop then runs zero iterations and
        returns final_output=None with no exception, which is a silent no-op
        rather than a loud failure. Flooring at one pass keeps that unreachable
        from this direction.
        """
        assert rounds_to_max_iter(rounds, 3) == 3

    @pytest.mark.parametrize("nodes", [0, -1])
    def test_non_positive_node_count_still_affords_one_iteration(self, nodes):
        assert rounds_to_max_iter(3, nodes) == 3

    def test_result_is_always_a_positive_int(self):
        for rounds in range(-3, 8):
            for nodes in range(-3, 8):
                got = rounds_to_max_iter(rounds, nodes)
                assert isinstance(got, int) and got >= 1


class TestStageWrappersAgreeWithTheSharedHelper:
    """Each stage keeps its own node count; none may re-derive the arithmetic."""

    def test_stage2_shard_wrapper_delegates(self):
        from src.orchestration.stage2.utils import (
            SHARD_GRAPH_NODE_COUNT,
            shard_rounds_to_max_iter,
        )

        for rounds in (1, 2, 3, 5):
            assert shard_rounds_to_max_iter(rounds) == rounds_to_max_iter(
                rounds, SHARD_GRAPH_NODE_COUNT
            )

    def test_stage3_generator_wrapper_delegates(self):
        from src.orchestration.stage3.extraction import (
            GENERATOR_GRAPH_NODE_COUNT,
            rounds_to_max_iter as stage3_rounds_to_max_iter,
        )

        for rounds in (1, 2, 3, 5):
            assert stage3_rounds_to_max_iter(rounds) == rounds_to_max_iter(
                rounds, GENERATOR_GRAPH_NODE_COUNT
            )

    def test_stage1_loops_are_whole_multiples_of_their_node_count(self):
        """A max_iter that is NOT a multiple of the node count stops part way
        through a round, returning a model no auditor ever saw."""
        from src.orchestration.stage1.loop_config import (
            ENRICHMENT_GRAPH_NODE_COUNT,
            EXTRACTION_GRAPH_NODE_COUNT,
            make_enrichment_loop_config,
            make_stage1_loop_config,
        )

        extraction = make_stage1_loop_config("some spec text")
        assert extraction.max_iter % EXTRACTION_GRAPH_NODE_COUNT == 0
        assert extraction.max_iter >= EXTRACTION_GRAPH_NODE_COUNT

        enrichment, _, _, _ = make_enrichment_loop_config([], [])
        assert enrichment.max_iter % ENRICHMENT_GRAPH_NODE_COUNT == 0
        assert enrichment.max_iter >= ENRICHMENT_GRAPH_NODE_COUNT

    def test_declared_node_counts_match_the_real_graphs(self):
        """The node count is the load-bearing input; if a node is added to a
        graph and its constant is not updated, the budget silently stops
        covering a whole round. Assert the constants against the real configs.
        """
        from src.orchestration.stage1.loop_config import (
            ENRICHMENT_GRAPH_NODE_COUNT,
            EXTRACTION_GRAPH_NODE_COUNT,
            make_enrichment_loop_config,
            make_stage1_loop_config,
        )

        extraction = make_stage1_loop_config("some spec text")
        assert len(extraction.agents) == EXTRACTION_GRAPH_NODE_COUNT

        enrichment, _, _, _ = make_enrichment_loop_config([], [])
        assert len(enrichment.agents) == ENRICHMENT_GRAPH_NODE_COUNT
