from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any

class ExplicitCardinality(BaseModel):
    """Extraction of row counts explicitly mentioned in facts, mapped to schema."""
    fact_id: int
    entity_name: str = Field(..., description="The name of the entity as described in the fact")
    target_table: Optional[str] = Field(None, description="The UPPER_SNAKE name of the matching table in the schema")
    approximate_count: Optional[int] = Field(None, description="The approximate integer value of the count if explicitly stateable")
    raw_count_description: str = Field(..., description="The original count description from the text (e.g., '2 million', '8-12')")
    context: str

class UnivariateKernel(BaseModel):
    """A pure statistical kernel extraction."""
    fact_id: int
    kernel_type: str = Field(..., description="e.g., Gamma, Weibull, Poisson, Zipf, Pareto, Mixture, Hurdle, Zero-Inflated")
    parameters: Dict[str, Any] = Field(..., description="Key-value pairs for distribution parameters. For mixtures, includes component probabilities.")
    description: str

class StateTransition(BaseModel):
    """Extraction of valid state machine paths and transition probabilities (e.g. Pending -> Paid)."""
    fact_id: int
    entity_name: str
    target_column: str = Field(..., description="The status or state column name")
    valid_states: List[str]
    transitions: Dict[str, List[str]] = Field(..., description="Mapping of source state to allowed destination states")
    logic: str

class EventProcess(BaseModel):
    """Arrival processes like Poisson or self-exciting Hawkes clusters."""
    fact_id: int
    process_type: str = Field(..., description="Poisson, Hawkes, Diurnal, Scheduled, Ornstein-Uhlenbeck, Random-Walk")
    event_table: str
    intensity_logic: str = Field(..., description="Description of event frequency or drift logic (e.g. '10 ops per sec', 'Mean-reverting drift to 0.5')")
    parameters: Dict[str, Any] = {}

class ConditionalPolicy(BaseModel):
    """Backward modeling cue where an external condition governs child behavior (e.g. Region Anomaly -> High Latency)."""
    fact_id: int
    condition_source: str = Field(..., description="The trigger (e.g. 'Region Degradation', 'Fraud Group')")
    affected_entities: List[str] = Field(..., description="List of tables/columns whose distribution changes under this condition")
    policy_logic: str = Field(..., description="Backward logic: 'If anomaly is present, mean latency increases 10x'")
    latent_variable_hint: str = Field(..., description="Suggestion for a Support Column (e.g. 'is_anomaly', 'user_tier')")

class StatisticalMapping(BaseModel):
    """Mapping of an extracted distribution or process to the refined schema."""
    fact_id: int
    target_table: str
    target_column: str
    mapping_logic: str = Field(..., description="Semantic reasoning for why this logic applies to this column")

class ShardDistributionMap(BaseModel):
    """The aggregate distribution map for a single shard."""
    shard_index: int
    cardinalities: List[ExplicitCardinality] = []
    kernels: List[UnivariateKernel] = []
    transitions: List[StateTransition] = []
    processes: List[EventProcess] = []
    policies: List[ConditionalPolicy] = []
    mappings: List[StatisticalMapping] = []

class SymbolicInvariant(BaseModel):
    """Algebraic or physical laws that must hold (e.g. Total = sum(Items), Velocity < Max)."""
    fact_id: int
    target_tables: List[str]
    invariant_type: str = Field(..., description="AlgebraicSum, VelocityLimit, ConservationLaw")
    logic: str = Field(..., description="The mathematical equality or inequality")

class GlobalDistributionRegistry(BaseModel):
    """The final consolidated data laws for the entire schema."""
    all_cardinalities: List[ExplicitCardinality]
    all_kernels: List[UnivariateKernel]
    all_transitions: List[StateTransition]
    all_processes: List[EventProcess]
    all_policies: List[ConditionalPolicy]
    all_mappings: List[StatisticalMapping]
    all_invariants: List[SymbolicInvariant] = []
