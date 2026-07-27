from pydantic import BaseModel, Field


class AblationConfig(BaseModel):
    enable_enrichment: bool = Field(
        True, description="Stage 1: run context_enricher to add external facts"
    )
    enable_sharding: bool = Field(
        True, description="Stage 2: shard facts into chunks for parallel architects"
    )
    enable_logical_constraints: bool = Field(
        True,
        description="Stage 4: enable IF-THEN logical constraint application (mask-based) in compiler",
    )
    use_bayesian_chunker: bool = Field(
        False,
        description=(
            "Stage 1: chunk facts with the Dirichlet-process Gibbs sampler "
            "(bayesian_chunker) instead of the default context-budget packer. "
            "The sampler is the paper's original method and is kept selectable "
            "for ablation, but it is NOT the default: measured on real runs it "
            "returned a single chunk for every input, at 12,000 sweeps of "
            "O(segments^2) each. See budget_chunker.py for the diagnosis."
        ),
    )

    @classmethod
    def full(cls) -> "AblationConfig":
        return cls(
            enable_enrichment=True,
            enable_sharding=True,
            enable_logical_constraints=True,
            use_bayesian_chunker=False,
        )

    @classmethod
    def bayesian_chunking(cls) -> "AblationConfig":
        """The pre-budget-packer behaviour, for a like-for-like comparison."""
        return cls(
            enable_enrichment=True,
            enable_sharding=True,
            enable_logical_constraints=True,
            use_bayesian_chunker=True,
        )

    @classmethod
    def no_enrichment(cls) -> "AblationConfig":
        return cls(
            enable_enrichment=False,
            enable_sharding=True,
            enable_logical_constraints=True,
            use_bayesian_chunker=False,
        )

    @classmethod
    def no_sharding(cls) -> "AblationConfig":
        return cls(
            enable_enrichment=True,
            enable_sharding=False,
            enable_logical_constraints=True,
            use_bayesian_chunker=False,
        )

    @classmethod
    def no_logical_constraints(cls) -> "AblationConfig":
        return cls(
            enable_enrichment=True,
            enable_sharding=True,
            enable_logical_constraints=False,
            use_bayesian_chunker=False,
        )
