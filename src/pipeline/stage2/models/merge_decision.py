from typing import List, Optional
from pydantic import BaseModel, Field


class TableMatchDecision(BaseModel):
    shard_a_table: str = Field(
        description="Table from the accumulating schema (shard A)."
    )
    shard_b_table: str = Field(description="Table from the incoming shard (shard B).")
    match_score: float = Field(description="Final GS match score.")
    name_score: float = Field(description="Name similarity (pre-modifier-penalty).")
    attr_score: float = Field(description="Column/attribute similarity score.")
    matched_columns: List[str] = Field(
        description="Column names from shard A that were matched."
    )
    shard_a_fact_ids: List[int] = Field(default_factory=list)
    shard_b_fact_ids: List[int] = Field(default_factory=list)
    had_pk_divergence: bool = Field(
        default=False, description="Tables had different PK names."
    )
    had_fk_target_divergence: bool = Field(
        default=False,
        description="Matched FK columns pointed to different target tables.",
    )


class UnmatchedTable(BaseModel):
    shard_b_table: str = Field(description="Table from shard B that was not matched.")
    reason: str = Field(
        description="'below_threshold', 'name_collision_merged', or 'new_entity'."
    )
    best_candidate_in_a: Optional[str] = Field(
        default=None, description="Best match candidate from shard A."
    )
    best_candidate_score: Optional[float] = Field(
        default=None, description="GS score for the best candidate."
    )
    shard_b_fact_ids: List[int] = Field(default_factory=list)


class MergeDecisionLog(BaseModel):
    matched_pairs: List[TableMatchDecision] = Field(default_factory=list)
    unmatched_tables: List[UnmatchedTable] = Field(default_factory=list)
    modifier_penalty_applied: List[str] = Field(
        default_factory=list,
        description="Table pairs ('tableA::tableB') where the distinct modifier penalty was applied.",
    )

    def __str__(self) -> str:
        lines = [
            f"MergeDecisionLog: {len(self.matched_pairs)} matched, "
            f"{len(self.unmatched_tables)} unmatched"
        ]
        if self.matched_pairs:
            lines.append("\nMatched Pairs:")
            for m in self.matched_pairs:
                flags = ""
                if m.had_pk_divergence:
                    flags += " [PK divergence]"
                if m.had_fk_target_divergence:
                    flags += " [FK target divergence]"
                lines.append(
                    f"  {m.shard_a_table} <-> {m.shard_b_table} "
                    f"(score={m.match_score:.2f}, name={m.name_score:.2f}, attr={m.attr_score:.2f}){flags}"
                )
                if m.matched_columns:
                    sample = ", ".join(m.matched_columns[:5])
                    lines.append(f"    cols: {sample}")
                if m.shard_a_fact_ids or m.shard_b_fact_ids:
                    lines.append(
                        f"    facts A={m.shard_a_fact_ids[:5]}, B={m.shard_b_fact_ids[:5]}"
                    )
        if self.unmatched_tables:
            lines.append("\nUnmatched Tables (from shard B):")
            for u in self.unmatched_tables:
                cand = ""
                if u.best_candidate_in_a:
                    cand = (
                        f" (best: {u.best_candidate_in_a}={u.best_candidate_score:.2f})"
                    )
                lines.append(f"  {u.shard_b_table} [{u.reason}]{cand}")
                if u.shard_b_fact_ids:
                    lines.append(f"    facts={u.shard_b_fact_ids[:5]}")
        if self.modifier_penalty_applied:
            lines.append(
                f"\nModifier penalties applied: {', '.join(self.modifier_penalty_applied)}"
            )
        return "\n".join(lines)
