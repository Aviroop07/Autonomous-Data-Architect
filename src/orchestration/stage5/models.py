from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class StrategyType(str, Enum):
    """
    Categorical strategy type for how a table's rows are generated.
    - INDEPENDENT_DIM: No FK dependencies. Generated first from thin air.
    - DEPENDENT_DIM: Has FK dependencies on other tables but is NOT a fact/event table.
    - FACT_SEQUENTIAL: Transactional/event rows ordered by time; children of dims.
    - FACT_NETWORK: Junction or relationship table connecting two or more entities (many-to-many).
    - SELF_REFERENTIAL: Has a FK back to its own primary key (e.g., employee.manager_id -> employee.employee_id).
    - SIMULATION_STATE: Requires iterative/stateful simulation (Markov chains, agent-based models, time-evolving state).
    - DERIVED_AGGREGATE: Table that is computed/aggregated from other tables (e.g., daily summary, leaderboard).
    - POLYMORPHIC_DISPATCH: Table whose FK points to one of several possible parent tables depending on a type discriminator column.
    """
    INDEPENDENT_DIM = "INDEPENDENT_DIM"
    DEPENDENT_DIM = "DEPENDENT_DIM"
    FACT_SEQUENTIAL = "FACT_SEQUENTIAL"
    FACT_NETWORK = "FACT_NETWORK"
    SELF_REFERENTIAL = "SELF_REFERENTIAL"
    SIMULATION_STATE = "SIMULATION_STATE"
    EVENT_PROCESS = "EVENT_PROCESS"             # Hawkes, Ornstein-Uhlenbeck, Random-Walk drift
    DERIVED_AGGREGATE = "DERIVED_AGGREGATE"
    POLYMORPHIC_DISPATCH = "POLYMORPHIC_DISPATCH"
    ALGEBRAIC_INVARIANT = "ALGEBRAIC_INVARIANT" # Gross = sum(Net) + Tax, Balance consistency


class StructuralFlag(str, Enum):
    """
    Optional flags that encode additional structural constraints for a table's generation.
    Multiple flags can apply to a single table.
    """
    BOOTSTRAP_REQUIRED = "BOOTSTRAP_REQUIRED"           # self-ref: generate a root subset with no parent first
    UNIQUE_COMPOSITE_KEY = "UNIQUE_COMPOSITE_KEY"       # bridge table: (fk_a, fk_b) pair must be unique
    TEMPORAL_ORDERING = "TEMPORAL_ORDERING"             # events must satisfy time constraints (e.g., cancel_date >= start_date)
    TYPE_DISCRIMINATOR = "TYPE_DISCRIMINATOR"           # polymorphic: table has a `type` column controlling FK target
    NULLABLE_FK = "NULLABLE_FK"                         # has at least one FK that is intentionally nullable
    ZIPF_DISTRIBUTION = "ZIPF_DISTRIBUTION"            # FK choices should follow a power-law (some parents are hot spots)
    TEMPORAL_VALIDITY_WINDOW = "TEMPORAL_VALIDITY_WINDOW"  # rows are only valid for a bounded time window
    LATENT_VARIABLE_GENERATION = "LATENT_VARIABLE_GENERATION" # table uses support columns to drive conditional distributions
    MULTI_PASS_SIMULATION = "MULTI_PASS_SIMULATION"    # requires a first pass to establish state, then a second pass for values
    MIXTURE_COMPONENT = "MIXTURE_COMPONENT"             # identifies a class/mixture membership column
    HURDLE_PROCESS = "HURDLE_PROCESS"                   # identifies a binary hurdle (zero-inflation) logic

class SupportColumn(BaseModel):
    """A temporary generative column used to drive conditional behavior but not existing in final SQL."""
    column_name: str = Field(..., description="e.g., is_anomaly, session_state, user_segment")
    description: str
    logic: str = Field(..., description="How this column is sampled before the real columns (e.g., '1% chance of TRUE')")

class ConditionalLogic(BaseModel):
    """Explicit mapping of how a latent column (support_column) affects one or more target columns."""
    support_column_name: str = Field(..., description="The name of the latent variable (SupportColumn) that drives the condition.")
    target_column_names: List[str] = Field(..., description="The columns in the table that are modified based on this condition.")
    effect_description: str = Field(..., description="Detailed description of the vectorized change (e.g., 'Increase latency by 5x using np.where').")

class TableGenerationStrategy(BaseModel):
    """Strategy for populating a single table in the correct order."""
    table_name: str = Field(..., description="UPPER_SNAKE_CASE table name from the global schema")
    order: int = Field(..., description="Integer generation order. 1 = first. Parent tables always have a lower order than their children.")
    strategy_type: StrategyType
    structural_flags: List[StructuralFlag] = Field(default_factory=list, description="Zero or more structural flags that complement the strategy type with additional generation constraints.")
    support_columns: List[SupportColumn] = Field(default_factory=list, description="Latent variables generated for this table to drive conditional business logic.")
    conditional_logics: List[ConditionalLogic] = Field(default_factory=list, description="Structured definitions of how support columns influence the final data.")
    target_row_count: int = Field(..., description="Estimated number of rows to generate for this table.")
    dependencies: List[str] = Field(default_factory=list, description="UPPER_SNAKE_CASE names of tables that must be generated before this one.")
    logic_summary: str = Field(..., description="Precise, step-by-step description of how to populate this table: which distributions to apply, how FKs are sampled, and how structural constraints (uniqueness, temporal ordering, bootstrapping) are enforced.")


class GenerationPlan(BaseModel):
    """The master plan for data synthesis."""
    ordered_tables: List[TableGenerationStrategy]
    total_expected_volume: int = Field(..., description="Sum of all target_row_count values across all tables.")
    generation_sequence_justification: str = Field(..., description="Concise explanation of why tables are ordered the way they are, calling out any non-obvious dependency decisions.")
