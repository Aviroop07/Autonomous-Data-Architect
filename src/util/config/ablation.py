from pydantic import BaseModel, Field


class AblationConfig(BaseModel):
    enable_enrichment: bool = Field(True, description="Stage 1: run context_enricher to add external facts")
    enable_sharding: bool = Field(True, description="Stage 2: shard facts into chunks for parallel architects")
    enable_logical_constraints: bool = Field(True, description="Stage 4: enable IF-THEN logical constraint application (mask-based) in compiler")

    @classmethod
    def full(cls) -> "AblationConfig":
        return cls(enable_enrichment=True, enable_sharding=True, enable_logical_constraints=True)

    @classmethod
    def no_enrichment(cls) -> "AblationConfig":
        return cls(enable_enrichment=False, enable_sharding=True, enable_logical_constraints=True)

    @classmethod
    def no_sharding(cls) -> "AblationConfig":
        return cls(enable_enrichment=True, enable_sharding=False, enable_logical_constraints=True)

    @classmethod
    def no_logical_constraints(cls) -> "AblationConfig":
        return cls(enable_enrichment=True, enable_sharding=True, enable_logical_constraints=False)
