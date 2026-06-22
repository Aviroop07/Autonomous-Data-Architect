import json
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.atomic_fact import FactTag
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.agents.chunker.agent import ChunkerLoopAgent
from src.pipeline.stage2.agents.domain_auditor.agent import DomainAuditorLoopAgent
from src.pipeline.stage2.agents.schema_architect.agent import SchemaArchitectLoopAgent
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.schema import Schema
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
)
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType, Severity
from src.util.schema_ops.patching_engine import apply_patches
from src.util.schema_ops.schema_patch import CritiqueReport


# ---------------------------------------------------------------------------
# Private types
# ---------------------------------------------------------------------------


class _ChunkerIssue(BaseModel):
    severity: Severity = Field(description="Severity level of the issue.")
    description: str = Field(description="Human-readable error description.")
    fact_id: Optional[int] = Field(default=None, description="Related fact ID, if any.")
    signature_key: Optional[str] = Field(
        default=None, description="Optional signature key for deduplication."
    )


def _format_chunk_errors(errors: List[_ChunkerIssue]) -> str:
    if not errors:
        return "No errors."
    lines = []
    for err in errors:
        sev_val = getattr(err, "severity", Severity.MEDIUM)
        sev = (
            sev_val.value.upper()
            if isinstance(sev_val, Severity)
            else str(sev_val).upper()
        )
        fact_id = getattr(err, "fact_id", None)
        description = getattr(err, "description", str(err))
        if fact_id is not None:
            lines.append(f"[{sev}] Fact {fact_id}: {description}")
        else:
            lines.append(f"[{sev}] {description}")
    return "\n".join(lines)


def validate_chunked_plan(
    plan: ChunkedPlan, facts: List[AtomicFact]
) -> List[ErrorRecord]:
    errors: List[ErrorRecord] = []
    input_ids = {f.id for f in facts}
    required_tags = {FactTag.STRUCTURAL, FactTag.LOGICAL, FactTag.STATISTICAL}
    required_ids = {f.id for f in facts if any(tag in required_tags for tag in f.tags)}

    core_ids = {f.id for f in plan.core_modeling_facts}
    missing_core = required_ids - core_ids
    if missing_core:
        errors.append(
            ErrorRecord(
                iteration=0,
                error_type=ErrorType.DETERMINISTIC,
                severity=Severity.CRITICAL,
                description=f"core_modeling_facts missing required fact IDs: {sorted(list(missing_core))}",
                signature_key="chunk_core_missing",
            )
        )

    extra_core = core_ids - input_ids
    if extra_core:
        errors.append(
            ErrorRecord(
                iteration=0,
                error_type=ErrorType.DETERMINISTIC,
                severity=Severity.CRITICAL,
                description=f"core_modeling_facts contains unknown fact IDs: {sorted(list(extra_core))}",
                signature_key="chunk_core_unknown",
            )
        )

    if not plan.chunks:
        errors.append(
            ErrorRecord(
                iteration=0,
                error_type=ErrorType.DETERMINISTIC,
                severity=Severity.CRITICAL,
                description="No chunks produced by chunker.",
                signature_key="chunk_none",
            )
        )
        return errors

    chunk_id_sets: List[set] = []
    for idx, chunk in enumerate(plan.chunks, start=1):
        if not chunk:
            errors.append(
                ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Chunk {idx} is empty.",
                    signature_key=f"chunk_empty:{idx}",
                )
            )
            chunk_id_sets.append(set())
            continue

        ids = {f.id for f in chunk}
        unknown_ids = ids - input_ids
        if unknown_ids:
            errors.append(
                ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Chunk {idx} contains unknown fact IDs: {sorted(list(unknown_ids))}",
                    signature_key=f"chunk_unknown:{idx}",
                )
            )
        chunk_id_sets.append(ids)

        if len(chunk) < 3:
            errors.append(
                ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.MEDIUM,
                    description=f"Chunk {idx} is very small ({len(chunk)} facts).",
                    signature_key=f"chunk_small:{idx}",
                )
            )
        elif len(chunk) > 20:
            errors.append(
                ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.MEDIUM,
                    description=f"Chunk {idx} is large ({len(chunk)} facts). Consider splitting.",
                    signature_key=f"chunk_large:{idx}",
                )
            )

    assigned_ids = set().union(*chunk_id_sets)
    missing_required = required_ids - assigned_ids
    if missing_required:
        errors.append(
            ErrorRecord(
                iteration=0,
                error_type=ErrorType.DETERMINISTIC,
                severity=Severity.CRITICAL,
                description=f"Required facts missing from all chunks: {sorted(list(missing_required))}",
                signature_key="chunk_missing_required",
            )
        )

    for idx, chunk in enumerate(plan.chunks, start=1):
        chunk_ids = chunk_id_sets[idx - 1] if idx - 1 < len(chunk_id_sets) else set()
        for fact in chunk:
            if not fact.referenced_fact_ids:
                continue
            for ref_id in fact.referenced_fact_ids:
                if ref_id in required_ids and ref_id not in chunk_ids:
                    errors.append(
                        ErrorRecord(
                            iteration=0,
                            error_type=ErrorType.DETERMINISTIC,
                            severity=Severity.HIGH,
                            description=(
                                f"Chunk {idx}: Fact {fact.id} references fact "
                                f"{ref_id} which is not in the same chunk."
                            ),
                            fact_id=fact.id,
                            signature_key=f"chunk_orphan_ref:{idx}:{fact.id}:{ref_id}",
                        )
                    )

    return errors


def repair_chunked_plan(plan: ChunkedPlan, facts: List[AtomicFact]) -> ChunkedPlan:
    fact_map = {f.id: f for f in facts}
    required_tags = {FactTag.STRUCTURAL, FactTag.LOGICAL, FactTag.STATISTICAL}
    required_ids = {f.id for f in facts if any(tag in required_tags for tag in f.tags)}

    seen_core: set = set()
    normalized_core: List[AtomicFact] = []
    for fact in plan.core_modeling_facts:
        if fact.id in fact_map and fact.id not in seen_core:
            normalized_core.append(fact_map[fact.id])
            seen_core.add(fact.id)
    for fid in sorted(required_ids):
        if fid not in seen_core and fid in fact_map:
            normalized_core.append(fact_map[fid])
            seen_core.add(fid)
    plan.core_modeling_facts = normalized_core

    reverse_refs: dict[int, set] = {f.id: set() for f in facts}
    for fact in facts:
        for ref_id in fact.referenced_fact_ids:
            if ref_id in reverse_refs:
                reverse_refs[ref_id].add(fact.id)

    new_chunks: List[List[AtomicFact]] = []
    for chunk in plan.chunks:
        chunk_ids: set = set()
        new_chunk: List[AtomicFact] = []
        for fact in chunk:
            if fact.id in fact_map and fact.id not in chunk_ids:
                new_chunk.append(fact_map[fact.id])
                chunk_ids.add(fact.id)
        new_chunks.append(new_chunk)

    if not new_chunks:
        new_chunks = [[]]

    assigned_ids = set().union(*[set(f.id for f in c) for c in new_chunks if c])
    missing_required = required_ids - assigned_ids

    for fid in sorted(missing_required):
        if fid not in fact_map:
            continue
        related = set(fact_map[fid].referenced_fact_ids) | reverse_refs.get(fid, set())
        best_idx = 0
        best_score = -1
        for idx, chunk in enumerate(new_chunks):
            chunk_ids = {f.id for f in chunk}
            score = len(related & chunk_ids)
            if score > best_score:
                best_score = score
                best_idx = idx
        new_chunks[best_idx].append(fact_map[fid])

    for idx, chunk in enumerate(new_chunks):
        chunk_ids = {f.id for f in chunk}
        to_add: List[AtomicFact] = []
        for fact in chunk:
            for ref_id in fact.referenced_fact_ids:
                if (
                    ref_id in required_ids
                    and ref_id not in chunk_ids
                    and ref_id in fact_map
                ):
                    to_add.append(fact_map[ref_id])
                    chunk_ids.add(ref_id)
        if to_add:
            chunk.extend(to_add)

    plan.chunks = new_chunks
    return plan


def _model_changed(current: BaseModel, prior: Optional[BaseModel]) -> Optional[bool]:
    if prior is None:
        return None
    return current.model_dump() != prior.model_dump()


# ---------------------------------------------------------------------------
# Loop output models for validator nodes
# ---------------------------------------------------------------------------


class _ChunkerValidationReport(LoopOutputModel):
    is_valid: bool = Field(description="True when no blocking errors remain.")
    errors: List[_ChunkerIssue] = Field(
        default_factory=list, description="Blocking errors for feedback."
    )
    plan: ChunkedPlan = Field(description="Repaired chunking plan.")

    def get_errors(self) -> list[str]:
        return [err.description for err in self.errors]


class _SchemaValidationReport(LoopOutputModel):
    is_valid: bool = Field(description="True when schema validation passes.")
    errors: List[str] = Field(default_factory=list, description="Validation errors.")

    def get_errors(self) -> list[str]:
        return self.errors


class _AuditPatchResult(LoopOutputModel):
    is_valid: bool = Field(description="True when patched schema validates.")
    had_patches: bool = Field(description="Whether auditor proposed any patches.")
    errors: List[str] = Field(
        default_factory=list, description="Validation errors after patching."
    )
    schema_state: Schema = Field(
        description="Schema to carry forward to the next iteration."
    )
    patched_schema: Optional[Schema] = Field(
        default=None,
        description="The patched schema (may be invalid when errors are present).",
    )

    def get_errors(self) -> list[str]:
        return self.errors


# ---------------------------------------------------------------------------
# LoopAgent subclasses for deterministic validator nodes
# ---------------------------------------------------------------------------


class ChunkerValidatorLoopAgent(LoopAgent):
    """Deterministic chunker validator node."""

    def __init__(self, facts: List[AtomicFact]) -> None:
        self._facts = facts

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        plan = ChunkedPlan.model_validate_json(query)
        plan = repair_chunked_plan(plan, self._facts)
        raw_errors = validate_chunked_plan(plan, self._facts)
        blocking = [
            e for e in raw_errors if e.severity in (Severity.CRITICAL, Severity.HIGH)
        ]
        issues = [
            _ChunkerIssue(
                severity=e.severity,
                description=e.description,
                fact_id=e.fact_id,
                signature_key=e.signature_key,
            )
            for e in blocking
        ]
        return _ChunkerValidationReport(
            is_valid=len(blocking) == 0,
            errors=issues,
            plan=plan,
        ), 0

    def build_context(self, ctx: LoopContext) -> str:
        chunker_output = ctx.node_outputs.get("chunker")
        if isinstance(chunker_output, ChunkedPlan):
            return chunker_output.model_dump_json(indent=2)
        return ChunkedPlan(core_modeling_facts=[], chunks=[]).model_dump_json(indent=2)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, _ChunkerValidationReport)
        changes_summary = (
            "valid" if output.is_valid else f"{len(output.errors)} blocking errors"
        )
        was_improvement = None
        if isinstance(prior, _ChunkerValidationReport):
            was_improvement = len(output.errors) < len(prior.errors)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )


class SchemaValidatorLoopAgent(LoopAgent):
    """Deterministic schema validator node."""

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        schema = Schema.model_validate_json(query)
        errors = schema._validate()
        return _SchemaValidationReport(is_valid=not errors, errors=errors), 0

    def build_context(self, ctx: LoopContext) -> str:
        schema_output = ctx.node_outputs.get("architect")
        if isinstance(schema_output, Schema):
            return schema_output.model_dump_json(indent=2)
        return Schema(tables=[], relationships=[]).model_dump_json(indent=2)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, _SchemaValidationReport)
        changes_summary = "valid" if output.is_valid else f"{len(output.errors)} errors"
        was_improvement = None
        if isinstance(prior, _SchemaValidationReport):
            was_improvement = len(output.errors) < len(prior.errors)
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )


class AuditPatchValidatorLoopAgent(LoopAgent):
    """Deterministic patch-apply + validate node.

    Holds a reference to DomainAuditorLoopAgent to read current_schema when
    building context without importing _AuditPatchResult into the auditor module.
    """

    def __init__(
        self,
        auditor: DomainAuditorLoopAgent,
        registry: Optional[TableFactRegistry],
        fact_clusters: Dict[str, List[AtomicFact]],
    ) -> None:
        self._auditor = auditor
        self._registry = registry
        self._owner_fact_ids: List[int] = list(
            {f.id for facts in fact_clusters.values() for f in facts}
        )
        self.fix_history: List[FixHistoryStep] = []
        self._attempt: int = 0

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        payload = json.loads(query) if query else {}
        schema = Schema(**payload.get("schema", {}))
        report = CritiqueReport(**payload.get("report", {}))

        if not report.patches:
            print("  [Auditor] No patches proposed.")
            return _AuditPatchResult(
                is_valid=True,
                had_patches=False,
                errors=[],
                schema_state=schema,
                patched_schema=None,
            ), 0

        print(f"  [Auditor] Applying {len(report.patches)} patch(es):")
        for p in report.patches:
            print(f"    {p}")

        temp_schema = schema.model_copy(deep=True)
        apply_patches(
            temp_schema,
            report.patches,
            registry=self._registry,
            owner_fact_ids=self._owner_fact_ids,
        )
        errors = temp_schema._validate()
        if not errors:
            print("  [Auditor] Schema valid after patches.")
            return _AuditPatchResult(
                is_valid=True,
                had_patches=True,
                errors=[],
                schema_state=temp_schema,
                patched_schema=temp_schema,
            ), 0

        print(f"  [Auditor] {len(errors)} validation error(s) after patches:")
        for e in errors[:5]:
            print(f"    - {e}")
        if len(errors) > 5:
            print(f"    ... ({len(errors) - 5} more)")

        self._attempt += 1
        self.fix_history.append(
            FixHistoryStep(
                attempt=self._attempt,
                errors=errors,
                corrections=[],
                fixed_schema=str(temp_schema),
                schema_state=temp_schema,
            )
        )
        return _AuditPatchResult(
            is_valid=False,
            had_patches=True,
            errors=errors,
            schema_state=schema,
            patched_schema=temp_schema,
        ), 0

    def build_context(self, ctx: LoopContext) -> str:
        report = ctx.node_outputs.get("auditor")
        current_schema = self._auditor.current_schema
        if isinstance(report, CritiqueReport):
            payload = {
                "schema": current_schema.model_dump(),
                "report": report.model_dump(),
            }
        else:
            payload = {
                "schema": current_schema.model_dump(),
                "report": {"patches": []},
            }
        return json.dumps(payload)

    def emit_history(
        self,
        output: LoopOutputModel,
        prior: Optional[LoopOutputModel],
        round_num: int,
        node: str,
    ) -> HistoryEntry:
        assert isinstance(output, _AuditPatchResult)
        was_improvement = None
        if isinstance(prior, _AuditPatchResult):
            was_improvement = len(output.errors) < len(prior.errors)
        if not output.had_patches:
            changes_summary = "no patches"
        elif output.is_valid:
            changes_summary = "patches applied"
        else:
            changes_summary = f"{len(output.errors)} errors after patch"
        return HistoryEntry(
            round=round_num,
            node=node,
            changes_summary=changes_summary,
            was_improvement=was_improvement,
        )


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


async def run_chunker_with_retry(
    facts: List[AtomicFact],
    max_retries: int = 5,
    model: Optional[str] = None,
    domain: Optional[str] = None,
    analytical_goal: Optional[str] = None,
) -> Tuple[ChunkedPlan, int]:
    chunker_agent = ChunkerLoopAgent(facts, domain, analytical_goal, model)
    validator_agent = ChunkerValidatorLoopAgent(facts)

    config = LoopConfig(
        agents={
            "chunker": AgentRoleConfig(
                agent_factory=lambda: chunker_agent,
                det_error_sources=["chunker_validator"],
            ),
            "chunker_validator": AgentRoleConfig(
                agent_factory=lambda: validator_agent,
            ),
        },
        graph={
            "edges": [
                GraphEdge(from_node="chunker", to_node="chunker_validator"),
                GraphEdge(
                    from_node="chunker_validator",
                    to_node="chunker",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="chunker_validator", to_node="end"),
            ]
        },
        start_node="chunker",
        max_iter=5,
    )

    result = await AgentLoop(config).run("")
    total_tokens = result.total_tokens

    report = result.node_outputs.get("chunker_validator")
    if isinstance(report, _ChunkerValidationReport):
        if report.is_valid:
            return report.plan, total_tokens
        print("    [Chunker] Exhausted retries. Falling back to a single chunk.")
        return ChunkedPlan(core_modeling_facts=facts, chunks=[facts]), total_tokens

    chunker_output = result.node_outputs.get("chunker")
    if isinstance(chunker_output, ChunkedPlan):
        return repair_chunked_plan(chunker_output, facts), total_tokens

    return ChunkedPlan(core_modeling_facts=facts, chunks=[facts]), total_tokens


async def run_architect_self_correction_loop(
    facts: List[AtomicFact],
    max_retries: int = 5,
    model: Optional[str] = None,
) -> Tuple[Schema, List[FixHistoryStep], int]:
    architect_agent = SchemaArchitectLoopAgent(facts, model)
    validator_agent = SchemaValidatorLoopAgent()

    config = LoopConfig(
        agents={
            "architect": AgentRoleConfig(
                agent_factory=lambda: architect_agent,
                det_error_sources=["schema_validator"],
            ),
            "schema_validator": AgentRoleConfig(
                agent_factory=lambda: validator_agent,
            ),
        },
        graph={
            "edges": [
                GraphEdge(from_node="architect", to_node="schema_validator"),
                GraphEdge(
                    from_node="schema_validator",
                    to_node="architect",
                    condition=EdgeCondition(field="is_valid", op="eq", value=False),
                ),
                GraphEdge(from_node="schema_validator", to_node="end"),
            ]
        },
        start_node="architect",
        max_iter=5,
    )

    result = await AgentLoop(config).run("")
    schema_output = result.node_outputs.get("architect")
    if not isinstance(schema_output, Schema):
        schema_output = Schema(tables=[], relationships=[])

    print(
        f"  [Architect] Shard output: {len(schema_output.tables)} tables, "
        f"{len(schema_output.relationships or [])} FKs, "
        f"{result.iteration_count} loop iteration(s)"
    )
    for t in schema_output.tables:
        col_str = ", ".join(f"{c.name}:{c.data_type or '?'}" for c in t.columns)
        print(f"    {t.name}  pk={t.pk}  [{col_str}]")
    for r in schema_output.relationships or []:
        print(
            f"    FK: {r.referencing_table}.{r.referencing_column} -> {r.referred_table}"
        )
    if architect_agent.fix_history:
        print(f"  [Architect] Retry iterations: {len(architect_agent.fix_history)}")
        for step in architect_agent.fix_history:
            print(f"    Attempt {step.attempt}: {len(step.errors)} error(s)")
            for e in step.errors[:3]:
                print(f"      - {e}")

    return schema_output, architect_agent.fix_history, result.total_tokens


async def run_auditor_self_correction_loop(
    shard_schema: Schema,
    intelligence: BaseModel,
    fact_clusters: Dict[str, List[AtomicFact]],
    max_retries: int = 5,
    registry: Optional[TableFactRegistry] = None,
    model: Optional[str] = None,
    initial_errors: Optional[List[str]] = None,
) -> Tuple[Schema, List[FixHistoryStep], int]:
    auditor_agent = DomainAuditorLoopAgent(
        schema=shard_schema,
        intelligence=intelligence,
        fact_clusters=fact_clusters,
        initial_errors=initial_errors,
        model=model,
    )
    patch_validator_agent = AuditPatchValidatorLoopAgent(
        auditor=auditor_agent,
        registry=registry,
        fact_clusters=fact_clusters,
    )

    config = LoopConfig(
        agents={
            "auditor": AgentRoleConfig(
                agent_factory=lambda: auditor_agent,
                det_error_sources=["patch_validator"],
            ),
            "patch_validator": AgentRoleConfig(
                agent_factory=lambda: patch_validator_agent,
            ),
        },
        graph={
            "edges": [
                GraphEdge(from_node="auditor", to_node="patch_validator"),
                GraphEdge(
                    from_node="patch_validator",
                    to_node="end",
                    condition=EdgeCondition(field="had_patches", op="eq", value=False),
                ),
                GraphEdge(
                    from_node="patch_validator",
                    to_node="end",
                    condition=EdgeCondition(field="is_valid", op="eq", value=True),
                ),
                GraphEdge(from_node="patch_validator", to_node="auditor"),
            ]
        },
        start_node="auditor",
        max_iter=5,
    )

    result = await AgentLoop(config).run("")
    patch_output = result.node_outputs.get("patch_validator")
    if isinstance(patch_output, _AuditPatchResult):
        shard_schema = patch_output.schema_state
    return shard_schema, patch_validator_agent.fix_history, result.total_tokens
