"""Unit tests for the Stage 3 Phase 2 bipartite variable-constraint graph
(src/util/algorithms/dof_graph.py). Deterministic, no LLM, no solver
randomness -- Dulmage-Mendelsohn decomposition is exact.
"""

from __future__ import annotations

import pytest

from src.util.algorithms.dof_graph import Constraint, DOFGraph, Variable


class TestVariableBounds:
    def test_bounds_default_to_none(self):
        assert Variable(name="x").lower_bound is None

    def test_accepts_ordered_bounds(self):
        v = Variable(name="x", lower_bound=1.0, upper_bound=8.0)
        assert (v.lower_bound, v.upper_bound) == (1.0, 8.0)

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lower_bound"):
            Variable(name="x", lower_bound=10.0, upper_bound=5.0)


class TestConstruction:
    def test_rejects_duplicate_variable_names(self):
        with pytest.raises(ValueError, match="Duplicate variable"):
            DOFGraph([Variable(name="x"), Variable(name="x")], [])

    def test_rejects_duplicate_constraint_names(self):
        variables = [Variable(name="x")]
        constraints = [
            Constraint(name="c", variables=["x"]),
            Constraint(name="c", variables=["x"]),
        ]
        with pytest.raises(ValueError, match="Duplicate constraint"):
            DOFGraph(variables, constraints)

    def test_rejects_name_collision_between_variable_and_constraint(self):
        variables = [Variable(name="shared")]
        constraints = [Constraint(name="shared", variables=["shared"])]
        with pytest.raises(ValueError, match="disjoint"):
            DOFGraph(variables, constraints)

    def test_rejects_constraint_referencing_undefined_variable(self):
        variables = [Variable(name="x")]
        constraints = [Constraint(name="c", variables=["y"])]
        with pytest.raises(ValueError, match="undefined variable"):
            DOFGraph(variables, constraints)

    def test_constraint_requires_at_least_one_variable(self):
        with pytest.raises(ValueError):
            Constraint(name="c", variables=[])


class TestClassification:
    def test_exactly_determined_and_loose_variables(self):
        """One constraint per variable pins it; an untouched variable is loose."""
        variables = [Variable(name="mu"), Variable(name="sigma"), Variable(name="free")]
        constraints = [
            Constraint(name="c_mu", variables=["mu"]),
            Constraint(name="c_sigma", variables=["sigma"]),
        ]
        result = DOFGraph(variables, constraints).classify()

        assert set(result.square_variables) == {"mu", "sigma"}
        assert result.loose_variables == ["free"]
        assert result.overconstrained_blocks == []

    def test_multi_variable_constraint_resolves_by_elimination(self):
        """A moment-target equation touching 3 variables, 2 of which are
        already independently pinned, must resolve the 3rd via elimination --
        this is the exact mechanism the real-data prototype validated for
        the line-items-per-order moment target (E[N_LI]~=1.5)."""
        variables = [Variable(name="a"), Variable(name="b"), Variable(name="c")]
        constraints = [
            Constraint(name="pin_a", variables=["a"]),
            Constraint(name="pin_b", variables=["b"]),
            Constraint(name="moment_target", variables=["a", "b", "c"]),
        ]
        result = DOFGraph(variables, constraints).classify()

        assert set(result.square_variables) == {"a", "b", "c"}
        assert result.loose_variables == []

    def test_conflicting_pins_form_one_overconstrained_block(self):
        """Two facts pinning the same variable must be reported as ONE block
        containing the variable AND both constraints -- not a single 'bad'
        constraint with the variable left looking clean. This is the
        block-level correction earned from the first failed toy experiment."""
        variables = [Variable(name="mu_ship"), Variable(name="mu_conflict")]
        constraints = [
            Constraint(name="c_ship", variables=["mu_ship"]),
            Constraint(name="c_conflict_a", variables=["mu_conflict"]),
            Constraint(name="c_conflict_b", variables=["mu_conflict"]),
        ]
        result = DOFGraph(variables, constraints).classify()

        assert result.square_variables == ["mu_ship"]
        assert len(result.overconstrained_blocks) == 1
        block = result.overconstrained_blocks[0]
        assert block.variables == ["mu_conflict"]
        assert set(block.constraints) == {"c_conflict_a", "c_conflict_b"}

    def test_isolated_variable_with_zero_edges_is_loose_not_overconstrained(self):
        variables = [Variable(name="isolated")]
        result = DOFGraph(variables, []).classify()

        assert result.loose_variables == ["isolated"]
        assert result.overconstrained_blocks == []

    def test_fact_references_are_preserved_on_input_objects(self):
        """Provenance isn't part of the classification output -- callers
        cross-reference loose/square/block names against their own
        Variable/Constraint list, so fact_references must survive untouched."""
        v = Variable(name="mu", fact_references=[42])
        graph = DOFGraph(
            [v], [Constraint(name="c", variables=["mu"], fact_references=[42, 43])]
        )

        assert graph.variables[0].fact_references == [42]
        assert graph.constraints[0].fact_references == [42, 43]
