from pathlib import Path
from typing import List, Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.merge_decision import MergeDecisionLog
from src.pipeline.stage2.models.schema import Schema
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response
from src.util.schema_ops.schema_patch import CritiqueReport

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(model: Optional[str] = None) -> AgentType:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=CritiqueReport,
        model=model,
        name="merge_reviewer",
    )


def _format_shard_facts(shard_facts: List[Tuple[int, List[AtomicFact]]]) -> str:
    lines = []
    for shard_idx, facts in shard_facts:
        lines.append(f"### Shard {shard_idx}")
        for f in facts:
            tags = ", ".join(f.tags) if f.tags else ""
            lines.append(f"  [{f.id}] {f.fact}" + (f" [{tags}]" if tags else ""))
    return "\n".join(lines)


def _format_shard_schemas(shard_schemas: List[Schema]) -> str:
    lines = []
    for i, shard in enumerate(shard_schemas):
        lines.append(f"### Shard {i + 1}")
        for t in shard.tables:
            cols = ", ".join(f"{c.name}:{c.data_type or 'VARCHAR'}" for c in t.columns)
            lines.append(f"  [{t.name}] pk={t.pk} | {cols}")
        for r in shard.relationships or []:
            lines.append(
                f"  FK: {r.referencing_table}.{r.referencing_column} -> {r.referred_table}"
            )
    return "\n".join(lines)


async def run_merge_review(
    merged_schema: Schema,
    decision_log: MergeDecisionLog,
    shard_facts: List[Tuple[int, List[AtomicFact]]],
    shard_schemas: Optional[List[Schema]] = None,
    agent: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[CritiqueReport, int]:
    if not agent:
        agent = get_agent(model)

    shard_schema_section = ""
    if shard_schemas:
        shard_schema_section = (
            f"## ORIGINAL SHARD SCHEMAS\n{_format_shard_schemas(shard_schemas)}\n\n"
        )

    query = (
        f"## MERGED SCHEMA\n{merged_schema.model_dump_json(indent=2)}\n\n"
        f"## MERGE DECISIONS\n{decision_log}\n\n"
        f"{shard_schema_section}"
        f"## SHARD FACTS\n{_format_shard_facts(shard_facts)}"
    )

    report, tokens = await get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query,
    )
    assert isinstance(report, CritiqueReport)
    return report, tokens
