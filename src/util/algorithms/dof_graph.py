"""Bipartite variable-constraint graph construction and Dulmage-Mendelsohn
feasibility classification for Stage 3 Phase 2.

See baselines/experiments/STAGE3_PHASE2_DESIGN.md section 2 for the design
this implements, and STAGE3_DOF_OPEN_QUESTIONS.md Q1/Q7/Q8 for the research
and hands-on validation behind it. Two rules here were real modeling mistakes
the first time this was prototyped, not stylistic choices:

1. A single NL fact stating two independent parameter values (e.g. "mean 8,
   std 2") must become two Constraints, one per parameter -- never one
   Constraint touching both Variables. One equation over two unknowns is
   mathematically underdetermined; this module cannot detect that mistake
   for a caller, since a genuine multi-variable relationship (e.g.
   subtotal = quantity * unit_price) legitimately needs one Constraint
   touching multiple Variables.
2. "Overconstrained" is a block-level classification, not a per-node
   verdict: a matched Constraint can still belong to a structurally
   overconstrained block alongside the unmatched Constraint that conflicts
   with it. classify() groups by connected component for exactly this
   reason -- do not report individual overconstrained nodes in isolation.
"""

from __future__ import annotations

import networkx as nx
from pydantic import BaseModel, Field, model_validator
from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn


class Variable(BaseModel):
    """A single scalar unknown: a distribution parameter, table row count, or
    FK fan-out mean. Never a whole distribution or a whole NL fact.

    lower_bound/upper_bound are domain-bound metadata (e.g. a stated row-count
    range), not graph edges -- per section 3's hybrid inequality handling, a
    bound never consumes a degree of freedom, it only narrows a variable that
    stays loose (or further restricts one that's already pinned). classify()
    ignores these fields entirely; they exist for downstream consumers (the
    LLM gap-filler, a later numeric-feasibility pass)."""

    name: str = Field(
        description="Unique identifier, e.g. 'mu_shipping_cost_nonplatinum'"
    )
    fact_references: list[int] = Field(
        default_factory=list,
        description="NL fact IDs that constrain this variable, if any",
    )
    lower_bound: float | None = Field(
        default=None, description="Inclusive lower domain bound, if stated"
    )
    upper_bound: float | None = Field(
        default=None, description="Inclusive upper domain bound, if stated"
    )

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Variable:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError(
                f"Variable '{self.name}': lower_bound ({self.lower_bound}) > upper_bound ({self.upper_bound})"
            )
        return self


class Constraint(BaseModel):
    """One equation, pinning or relating a specific set of Variables."""

    name: str = Field(description="Unique identifier")
    variables: list[str] = Field(
        min_length=1,
        description="Names of the Variables this constraint's equation touches",
    )
    fact_references: list[int] = Field(
        default_factory=list,
        description="NL fact IDs this constraint derives from",
    )


class OverconstrainedBlock(BaseModel):
    """A connected group of Constraints and Variables where more equations
    touch the group than it has degrees of freedom to satisfy them. May be
    harmless redundancy (two facts agreeing) or a genuine contradiction --
    telling those apart needs a numeric check, not just this structural
    classification (STAGE3_PHASE2_DESIGN.md section 3)."""

    variables: list[str]
    constraints: list[str]


class DOFClassification(BaseModel):
    """Structural feasibility result for one DOFGraph."""

    square_variables: list[str] = Field(description="Exactly-determined variables")
    loose_variables: list[str] = Field(
        description="Free variables with no value yet -- an LLM-probe target"
    )
    overconstrained_blocks: list[OverconstrainedBlock] = Field(
        description="Connected components flagged as structurally overconstrained"
    )


class DOFGraph:
    """Bipartite Variable-Constraint graph, classified via Dulmage-Mendelsohn
    decomposition (pyomo.contrib.incidence_analysis)."""

    def __init__(
        self, variables: list[Variable], constraints: list[Constraint]
    ) -> None:
        var_names = [v.name for v in variables]
        con_names = [c.name for c in constraints]
        if len(var_names) != len(set(var_names)):
            raise ValueError("Duplicate variable names")
        if len(con_names) != len(set(con_names)):
            raise ValueError("Duplicate constraint names")
        if collision := set(var_names) & set(con_names):
            raise ValueError(
                f"Variable and constraint names must be disjoint: {sorted(collision)}"
            )

        known = set(var_names)
        for c in constraints:
            if unknown := set(c.variables) - known:
                raise ValueError(
                    f"Constraint '{c.name}' references undefined variable(s): {sorted(unknown)}"
                )

        self.variables = variables
        self.constraints = constraints
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from((v.name for v in self.variables), bipartite=0)
        graph.add_nodes_from((c.name for c in self.constraints), bipartite=1)
        for c in self.constraints:
            for v in c.variables:
                graph.add_edge(c.name, v)
        return graph

    def classify(self) -> DOFClassification:
        top_nodes = [c.name for c in self.constraints]
        row_part, col_part = dulmage_mendelsohn(self.graph, top_nodes=top_nodes)

        loose = set(col_part.unmatched) | set(col_part.underconstrained)
        # row_part.unmatched holds the "losing" side of a redundant pair (e.g.
        # two facts pinning the same variable) -- it belongs in the same
        # block as the constraint that DID match, so it must be included here
        # even though it never appears in row_part.overconstrained itself.
        block_nodes = (
            set(row_part.unmatched)
            | set(row_part.overconstrained)
            | set(col_part.overconstrained)
        )

        blocks = [
            OverconstrainedBlock(
                variables=sorted(n for n in component if n in self._variable_names),
                constraints=sorted(n for n in component if n in self._constraint_names),
            )
            for component in nx.connected_components(self.graph.subgraph(block_nodes))
        ]

        return DOFClassification(
            square_variables=sorted(col_part.square),
            loose_variables=sorted(loose),
            overconstrained_blocks=blocks,
        )

    @property
    def _variable_names(self) -> set[str]:
        return {v.name for v in self.variables}

    @property
    def _constraint_names(self) -> set[str]:
        return {c.name for c in self.constraints}
