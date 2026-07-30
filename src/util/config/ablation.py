from typing import Optional

from pydantic import BaseModel, Field


class AblationConfig(BaseModel):
    enable_enrichment: bool = Field(
        default=True, description="Stage 1: run context_enricher to add external facts"
    )
    enable_sharding: bool = Field(
        default=True,
        description="Stage 2: shard facts into chunks for parallel architects",
    )
    enable_logical_constraints: bool = Field(
        default=True,
        description="Stage 4: enable IF-THEN logical constraint application (mask-based) in compiler",
    )
    use_bayesian_chunker: bool = Field(
        default=False,
        description=(
            "Stage 1: chunk facts with the Dirichlet-process Gibbs sampler "
            "(bayesian_chunker) instead of the default context-budget packer. "
            "The sampler is the paper's original method and is kept selectable "
            "for ablation, but it is NOT the default: measured on real runs it "
            "returned a single chunk for every input, at 12,000 sweeps of "
            "O(segments^2) each. See budget_chunker.py for the diagnosis."
        ),
    )
    chunk_budget_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Stage 1: override the per-chunk token budget instead of deriving it "
            "from the model's live-queried context window. Exists because the "
            "real budget makes multi-chunk extraction UNREACHABLE, and therefore "
            "Stage 2's shard-and-merge unreachable with it: the most complex "
            "saved run produced 88 facts totalling 1,963 fact-tokens against a "
            "~623,000-token budget, a factor of 317, so a genuine second chunk "
            "would need a specification yielding roughly 28,000 facts. Setting a "
            "small budget here is the only way to exercise the parallel "
            "extract-and-merge path at all, so it is an ablation knob rather "
            "than a tuning parameter -- leave it None for a faithful run."
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
    def forced_multi_chunk(cls, chunk_budget_tokens: int) -> "AblationConfig":
        """Shrink the per-chunk budget until extraction genuinely shards.

        No longer the ONLY way to reach shard-and-merge, and the claim that used
        to stand here -- that the context budget exceeds real fact volume by
        ~317x so every faithful run is single-chunk -- described a defect rather
        than a property of the inputs. BudgetChunker now also bounds a chunk by
        per-call extraction CAPACITY, so a large-schema spec shards on the
        faithful path too.

        This knob remains useful for two things: setting a budget the capacity
        default would not choose (calibration sweeps), and reproducing the old
        single-chunk behaviour for comparison. A budget set here is still
        ARTIFICIAL, so a result obtained under it is an ablation measurement.

        For reference, on the 88-fact / 1,963-token complex run: 981 tokens
        yields 3 chunks, 490 yields 5, and 245 yields 9.
        """
        return cls(
            enable_enrichment=True,
            enable_sharding=True,
            enable_logical_constraints=True,
            use_bayesian_chunker=False,
            chunk_budget_tokens=chunk_budget_tokens,
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
