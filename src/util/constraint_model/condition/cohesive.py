"""The three cohesive Condition terms: Distributed, Correlated, StateSequence
(Sections 8-9). Cohesive because their parameters only mean something
jointly -- unlike a bare moment fact (Section 8.3), which always decomposes
into an ordinary Aggregate + Comparison instead of a dedicated node here.

**Standalone-only rule (Section 9.3)**: none of these three can ever nest
inside RAnd/ROr/RIfThen with ordinary predicates or each other. This is
enforced by construction, not by a runtime check -- none of them are
members of predicates.py's RPredicateUnion, so they are structurally
impossible to place inside an RAnd/ROr/RIfThen operand. They only ever
appear as the sole, top-level Condition of their own Constraint
(constraint.py, task #30).

**Deliberately NOT this module's job**:
- Column-type compatibility against the real synthesized Relation schema
  (Distributed's family needs a numeric/categorical column;
  StateSequence.sequence_column must be discrete/categorical) -- needs
  external schema context, belongs in condition/validate.py (task #29),
  same reasoning as every other external-context check deferred so far.
- Cross-fact consistency (Section 6, 8.1's family-mismatch handling,
  9.2's transition-graph merge/cycle detection) -- these all compare
  MULTIPLE Constraints against each other, not one term's own structural
  validity. That is reconciliation-layer work, out of scope here.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field

_LOWER_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_lower_snake(name: str) -> bool:
    return bool(_LOWER_SNAKE_RE.fullmatch(name))


def _as_float(value: Union[float, List[str], List[float]]) -> Union[float, None]:
    """Narrows a Distributed.parameters value to a float, or None if it
    isn't one (a malformed value here is reported as an "unknown keys"-
    style structural issue elsewhere, not silently coerced)."""
    return value if isinstance(value, (int, float)) else None


DistributionFamily = Literal[
    "GAUSSIAN", "LOG_NORMAL", "BETA", "POISSON", "CATEGORICAL", "UNIFORM"
]

_REQUIRED_DISTRIBUTION_PARAMS: Dict[str, frozenset[str]] = {
    "GAUSSIAN": frozenset({"mean", "std_dev"}),
    "LOG_NORMAL": frozenset({"mean", "std_dev"}),
    "BETA": frozenset({"alpha", "beta"}),
    "POISSON": frozenset({"lam"}),
    "CATEGORICAL": frozenset({"categories"}),
    "UNIFORM": frozenset({"min_value", "max_value"}),
}
_ALLOWED_EXTRA_DISTRIBUTION_PARAMS: Dict[str, frozenset[str]] = {
    "CATEGORICAL": frozenset({"probabilities"}),
}


class Distributed(BaseModel):
    """A column's marginal distribution: family + PARTIAL parameters
    (Section 8.1). Missing parameters become free/loose DOF variables for
    Stage 4 -- only the PRESENT parameters are domain-checked here; a
    parameter being absent is never itself an error."""

    node_type: Literal["distributed"] = "distributed"
    column: str = Field(description="The column this distribution applies to.")
    family: DistributionFamily
    parameters: Dict[str, Union[float, List[str], List[float]]] = Field(
        default_factory=dict, description="Family-specific parameters, may be partial."
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not _is_lower_snake(self.column):
            errors.append(
                f"Distributed.column must be lower_snake_case, got '{self.column}'."
            )

        allowed = _REQUIRED_DISTRIBUTION_PARAMS.get(
            self.family, frozenset()
        ) | _ALLOWED_EXTRA_DISTRIBUTION_PARAMS.get(self.family, frozenset())
        unknown = set(self.parameters) - allowed
        if unknown:
            errors.append(
                f"Distributed.parameters contains keys not recognized for "
                f"{self.family}: {sorted(unknown)}."
            )

        params = self.parameters
        std_dev = _as_float(params["std_dev"]) if "std_dev" in params else None
        if (
            self.family in ("GAUSSIAN", "LOG_NORMAL")
            and std_dev is not None
            and std_dev <= 0
        ):
            errors.append(f"Distributed({self.family}).std_dev must be positive.")
        if self.family == "BETA":
            alpha = _as_float(params["alpha"]) if "alpha" in params else None
            beta = _as_float(params["beta"]) if "beta" in params else None
            if alpha is not None and alpha <= 0:
                errors.append("Distributed(BETA).alpha must be positive.")
            if beta is not None and beta <= 0:
                errors.append("Distributed(BETA).beta must be positive.")
        if self.family == "POISSON":
            lam = _as_float(params["lam"]) if "lam" in params else None
            if lam is not None and lam <= 0:
                errors.append("Distributed(POISSON).lam must be positive.")
        if self.family == "CATEGORICAL" and "categories" in params:
            cats = params["categories"]
            if not isinstance(cats, list) or len(cats) == 0:
                errors.append(
                    "Distributed(CATEGORICAL).categories must be a non-empty list."
                )
            probs = params.get("probabilities")
            if isinstance(probs, list) and isinstance(cats, list):
                numeric_probs = [p for p in probs if isinstance(p, (int, float))]
                if len(numeric_probs) != len(probs):
                    errors.append(
                        "Distributed(CATEGORICAL).probabilities must all be numeric."
                    )
                elif len(numeric_probs) != len(cats):
                    errors.append(
                        "Distributed(CATEGORICAL).probabilities length must match categories."
                    )
                elif not math.isclose(sum(numeric_probs), 1.0, rel_tol=1e-5):
                    errors.append(
                        "Distributed(CATEGORICAL).probabilities must sum to 1.0."
                    )
                elif any(p < 0 for p in numeric_probs):
                    errors.append(
                        "Distributed(CATEGORICAL).probabilities cannot be negative."
                    )
        if self.family == "UNIFORM":
            min_value = (
                _as_float(params["min_value"]) if "min_value" in params else None
            )
            max_value = (
                _as_float(params["max_value"]) if "max_value" in params else None
            )
            if (
                min_value is not None
                and max_value is not None
                and min_value > max_value
            ):
                errors.append("Distributed(UNIFORM).min_value must be <= max_value.")
        return errors


CorrelationFamily = Literal["GAUSSIAN", "STUDENT_T", "CLAYTON", "GUMBEL", "FRANK"]


class PairwiseCorrelation(BaseModel):
    """One partially-specified entry of a Correlated term's implied
    correlation matrix. Omitting a pair entirely leaves it a free
    variable for Stage 4 -- this is how Correlated's parameters stay
    partial (Section 8.2)."""

    left: str = Field(description="One of Correlated.columns.")
    right: str = Field(description="Another of Correlated.columns.")
    value: float = Field(description="Correlation value in [-1, 1].")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not (-1.0 <= self.value <= 1.0):
            errors.append(
                f"PairwiseCorrelation.value must be in [-1, 1], got {self.value}."
            )
        if self.left == self.right:
            errors.append(
                "PairwiseCorrelation: left and right must be different columns."
            )
        return errors


class Correlated(BaseModel):
    """Joint dependence across an arbitrary-arity column list (Section
    8.2), covering numeric-numeric, categorical-categorical (polychoric),
    and mixed (polyserial) dependence via ONE mechanism -- every pairwise
    entry is an ordinary correlation value in a shared latent space
    (Section 8.2.1), regardless of the two columns' own types.
    `shared_parameters` holds family-wide values shared across every
    column in the joint family (e.g. STUDENT_T's one shared `nu`)."""

    node_type: Literal["correlated"] = "correlated"
    columns: List[str] = Field(min_length=2, description="The joint column set.")
    family: CorrelationFamily
    pairwise: List[PairwiseCorrelation] = Field(
        default_factory=list, description="Partial pairwise correlations."
    )
    shared_parameters: Dict[str, float] = Field(
        default_factory=dict,
        description="Family-wide shared parameters, e.g. STUDENT_T's nu.",
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if len(self.columns) != len(set(self.columns)):
            errors.append("Correlated.columns contains duplicate column names.")
        for c in self.columns:
            if not _is_lower_snake(c):
                errors.append(
                    f"Correlated.columns entry must be lower_snake_case, got '{c}'."
                )

        col_set = set(self.columns)
        for i, pw in enumerate(self.pairwise):
            errors.extend(f"Correlated.pairwise[{i}]: {e}" for e in pw._validate())
            if pw.left not in col_set:
                errors.append(
                    f"Correlated.pairwise[{i}].left '{pw.left}' not in Correlated.columns."
                )
            if pw.right not in col_set:
                errors.append(
                    f"Correlated.pairwise[{i}].right '{pw.right}' not in Correlated.columns."
                )
        return errors


class StateTransition(BaseModel):
    """One directed edge in a StateSequence's transition graph."""

    from_state: str = Field(description="The sequence_column value transitioned from.")
    to_state: str = Field(description="The sequence_column value transitioned to.")

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if self.from_state == self.to_state:
            errors.append(
                "StateTransition: from_state and to_state must differ (a self-loop is "
                "not a transition)."
            )
        return errors


class StateSequence(BaseModel):
    """State-machine fact over a single categorical column's value (Section
    9.1, revised) -- e.g. an order's status must follow
    ready -> packed -> shipped -> delivered. This is a transition-graph
    invariant on the column's CURRENT value, not a window/ordering claim
    over multiple rows -- no event-log/history table is assumed to exist.
    Cross-fact consistency (grouping facts by sequence_column + population,
    merging transition graphs, cycle detection) is Section 9.2's algorithm,
    NOT implemented here -- this is one fact's own structural validity
    only."""

    node_type: Literal["state_sequence"] = "state_sequence"
    sequence_column: str = Field(description="The categorical column tracked as state.")
    allowed_transitions: List[StateTransition] = Field(default_factory=list)
    forbidden_transitions: List[StateTransition] = Field(default_factory=list)
    strict: bool = Field(
        default=False,
        description=(
            "If True, this fact asserts the sequence is acyclic -- a cycle in the "
            "merged allowed-transitions graph across all facts sharing this "
            "table/sequence_column becomes a conflict (Section 9.2 step 4). Cycles "
            "are allowed by default."
        ),
    )

    def _validate(self) -> List[str]:
        errors: List[str] = []
        if not _is_lower_snake(self.sequence_column):
            errors.append(
                f"StateSequence.sequence_column must be lower_snake_case, got "
                f"'{self.sequence_column}'."
            )
        for i, t in enumerate(self.allowed_transitions):
            errors.extend(
                f"StateSequence.allowed_transitions[{i}]: {e}" for e in t._validate()
            )
        for i, t in enumerate(self.forbidden_transitions):
            errors.extend(
                f"StateSequence.forbidden_transitions[{i}]: {e}" for e in t._validate()
            )

        allowed_set = {(t.from_state, t.to_state) for t in self.allowed_transitions}
        forbidden_set = {(t.from_state, t.to_state) for t in self.forbidden_transitions}
        conflict = allowed_set & forbidden_set
        if conflict:
            errors.append(
                f"StateSequence: transition(s) {sorted(conflict)} are asserted both "
                "allowed and forbidden within this same fact."
            )
        return errors


CohesiveUnion = Annotated[
    Union[Distributed, Correlated, StateSequence], Field(discriminator="node_type")
]
