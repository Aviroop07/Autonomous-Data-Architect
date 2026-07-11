from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
import math


class ConstraintBase(BaseModel):
    fact_references: List[int] = Field(
        min_items=1,
        description="The exact Stage 1 Fact IDs that explicitly state this rule. MUST NOT BE EMPTY.",
    )


class TableCardinality(ConstraintBase):
    table_name: str = Field(description="Name of the table in UPPER_SNAKE_CASE.")
    min_rows: int = Field(description="Inclusive lower bound on row count.")
    max_rows: int = Field(description="Inclusive upper bound on row count.")

    def _validate(self) -> List[str]:
        errors = []
        if self.min_rows < 0:
            errors.append(
                f"TableCardinality {self.table_name}: min_rows cannot be negative."
            )
        if self.max_rows < self.min_rows:
            errors.append(
                f"TableCardinality {self.table_name}: max_rows ({self.max_rows}) < min_rows ({self.min_rows})."
            )
        return errors


class FanoutConstraint(ConstraintBase):
    parent_table: str = Field(description="Parent table in UPPER_SNAKE_CASE.")
    child_table: str = Field(description="Child table containing the foreign key.")
    foreign_key_columns: List[str] = Field(
        description="Foreign key column(s) in the child table."
    )
    min_fanout: float = Field(description="Minimum average children per parent.")
    max_fanout: Optional[float] = Field(
        default=None, description="Maximum average children per parent."
    )

    def _validate(self) -> List[str]:
        errors = []
        if self.min_fanout < 0:
            errors.append(
                f"Fanout {self.parent_table}->{self.child_table}: min_fanout cannot be negative."
            )
        if self.max_fanout is not None and self.max_fanout < self.min_fanout:
            errors.append(
                f"Fanout {self.parent_table}->{self.child_table}: max_fanout < min_fanout."
            )
        return errors


class UniqueConstraint(ConstraintBase):
    table_name: str = Field(description="Table containing the unique columns.")
    columns: List[str] = Field(
        min_items=1, description="List of columns that form a unique composite key."
    )

    def _validate(self) -> List[str]:
        return []


class AggregationConstraint(ConstraintBase):
    parent_table: str = Field(description="Table holding the aggregated sum/avg.")
    parent_column: str = Field(description="Column holding the aggregated result.")
    descendant_table: str = Field(
        description="Table holding the raw values to be aggregated (can be a grandchild)."
    )
    descendant_column: str = Field(description="Column containing the raw values.")
    operation: Literal["SUM", "AVG", "MAX", "MIN"] = Field(
        description="The mathematical rollup operation."
    )

    def _validate(self) -> List[str]:
        return []


class DistributionBase(ConstraintBase):
    table_name: str = Field(description="Table containing the column.")
    column_name: str = Field(description="Column name.")
    if_condition: Optional[str] = Field(
        default=None,
        description="Condition restricting when this distribution applies.",
    )

    def _validate(self) -> List[str]:
        errors = []
        if self.if_condition is not None and not self.if_condition.strip():
            errors.append(
                f"DistributionBase {self.column_name}: if_condition cannot be empty whitespace."
            )
        return errors


class GaussianDistribution(DistributionBase):
    family: Literal["GAUSSIAN"] = Field("GAUSSIAN")
    mean: float
    std_dev: float

    def _validate(self) -> List[str]:
        errors = []
        if self.std_dev <= 0:
            errors.append(f"Gaussian {self.column_name}: std_dev must be positive.")
        return errors


class LogNormalDistribution(DistributionBase):
    family: Literal["LOG_NORMAL"] = Field("LOG_NORMAL")
    mean: float
    std_dev: float

    def _validate(self) -> List[str]:
        errors = []
        if self.std_dev <= 0:
            errors.append(f"LogNormal {self.column_name}: std_dev must be positive.")
        return errors


class BetaDistribution(DistributionBase):
    family: Literal["BETA"] = Field("BETA")
    alpha: float
    beta: float

    def _validate(self) -> List[str]:
        errors = []
        if self.alpha <= 0 or self.beta <= 0:
            errors.append(f"Beta {self.column_name}: alpha and beta must be positive.")
        return errors


class PoissonDistribution(DistributionBase):
    family: Literal["POISSON"] = Field("POISSON")
    lam: float = Field(description="Lambda parameter (expected rate/mean).")

    def _validate(self) -> List[str]:
        errors = []
        if self.lam <= 0:
            errors.append(f"Poisson {self.column_name}: lambda must be positive.")
        return errors


class CategoricalDistribution(DistributionBase):
    family: Literal["CATEGORICAL"] = Field("CATEGORICAL")
    categories: List[str]
    probabilities: Optional[List[float]] = Field(default=None)

    def _validate(self) -> List[str]:
        errors = []
        if not self.categories:
            errors.append(
                f"Categorical {self.column_name}: categories cannot be empty."
            )
        if self.probabilities is not None:
            if len(self.categories) != len(self.probabilities):
                errors.append(
                    f"Categorical {self.column_name}: categories and probabilities length mismatch."
                )
            if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-5):
                errors.append(
                    f"Categorical {self.column_name}: probabilities must sum to 1.0."
                )
            if any(p < 0 for p in self.probabilities):
                errors.append(
                    f"Categorical {self.column_name}: probabilities cannot be negative."
                )
        return errors


class UniformDistribution(DistributionBase):
    family: Literal["UNIFORM"] = Field("UNIFORM")
    min_value: float
    max_value: float

    def _validate(self) -> List[str]:
        errors = []
        if self.min_value > self.max_value:
            errors.append(f"Uniform {self.column_name}: min_value > max_value.")
        return errors


DistributionConstraint = Union[
    GaussianDistribution,
    LogNormalDistribution,
    BetaDistribution,
    PoissonDistribution,
    CategoricalDistribution,
    UniformDistribution,
]


class FormatConstraint(ConstraintBase):
    table_name: str = Field(description="Table name.")
    column_name: str = Field(description="Column name.")
    regex_pattern: Optional[str] = Field(
        default=None, description="Explicit regex pattern if specified."
    )
    semantic_type: Optional[str] = Field(
        default=None, description="Faker provider (e.g. 'email', 'phone_number')."
    )

    def _validate(self) -> List[str]:
        errors = []
        if not self.regex_pattern and not self.semantic_type:
            errors.append(
                f"FormatConstraint {self.column_name}: Must provide either regex_pattern or semantic_type."
            )
        return errors


class CrossColumnLogic(ConstraintBase):
    table_context: str = Field(
        description="The primary table or a JOIN clause defining the evaluation scope."
    )
    if_condition: Optional[str] = Field(
        default=None, description="The IF condition in standard SQL syntax."
    )
    then_enforcement: str = Field(description="The THEN enforcement in SQL syntax.")

    def _validate(self) -> List[str]:
        errors = []
        if not self.table_context.strip():
            errors.append("CrossColumnLogic: table_context cannot be empty.")
        if not self.then_enforcement.strip():
            errors.append("CrossColumnLogic: then_enforcement cannot be empty.")
        if self.if_condition is not None and not self.if_condition.strip():
            errors.append("CrossColumnLogic: if_condition cannot be empty whitespace.")
        return errors


class MomentTarget(ConstraintBase):
    table_name: str = Field(description="Table containing the column.")
    column_name: str = Field(
        description="Column the statistic describes -- often itself derived (e.g. an aggregated total), not a base column with its own DistributionConstraint."
    )
    statistic: Literal["MEAN"] = Field(
        description="Which population statistic is being pinned. MEAN only for now -- see STAGE3_PHASE2_DESIGN.md section 4.4."
    )
    target_value: float = Field(description="The stated value of the statistic.")

    def _validate(self) -> List[str]:
        return []


class StatisticalManifest(BaseModel):
    distributions: List[DistributionConstraint] = Field(default_factory=list)
    moment_targets: List[MomentTarget] = Field(default_factory=list)


class StructuralManifest(BaseModel):
    cardinalities: List[TableCardinality] = Field(default_factory=list)
    fanouts: List[FanoutConstraint] = Field(default_factory=list)
    uniqueness: List[UniqueConstraint] = Field(default_factory=list)
    aggregations: List[AggregationConstraint] = Field(default_factory=list)


class LogicManifest(BaseModel):
    formats: List[FormatConstraint] = Field(default_factory=list)
    cross_column_logic: List[CrossColumnLogic] = Field(default_factory=list)


class ConstraintManifest(BaseModel):
    statistical: StatisticalManifest = Field(default_factory=StatisticalManifest)
    structural: StructuralManifest = Field(default_factory=StructuralManifest)
    logic: LogicManifest = Field(default_factory=LogicManifest)
