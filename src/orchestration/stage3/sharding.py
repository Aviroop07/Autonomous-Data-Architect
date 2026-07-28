"""Deriving schema shards when the caller does not supply them.

Builds the ILP's inputs from the Stage 2 schema plus the Stage 1 fact set and
turns its answer back into real Schema objects. None of the ILP's
hyperparameters are exposed to callers -- see _derive_shards_from_schema.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.registry import TableFactRegistry
from src.util.schema_model.schema import Schema, Table
from src.pipeline.stage3.middleware.fact_allocation import find_mentioned_tables
from src.pipeline.stage3.middleware.fact_column_mapping import build_fact_column_map
from src.util.algorithms.sharding_ilp import shard_schema_auto
from src.util.core.agent import _detect_provider

logger = logging.getLogger(__name__)


def _build_registry(facts: List[AtomicFact], schema: Schema) -> TableFactRegistry:
    registry = TableFactRegistry()
    table_names = [t.name for t in schema.tables]
    for fact in facts:
        mentioned = find_mentioned_tables(fact.fact, table_names)
        for table_name in mentioned:
            registry.register_table_facts(table_name, [fact.id])
    return registry


def _build_shard_table_sets(shards: List[Schema]) -> List[set[str]]:
    return [{t.name for t in shard.tables} for shard in shards]


_MAX_MENTIONED_TABLES_FOR_SHARDING = 4


def _build_ilp_inputs(
    schema: Schema, facts: List[AtomicFact]
) -> Tuple[
    List[str],
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[Tuple[str, str, str, str]],
    Dict[str, List[Tuple[str, str]]],
]:
    tables = [t.name for t in schema.tables]
    columns_by_table = {t.name: [c.name for c in t.columns] for t in schema.tables}
    pks_by_table = {t.name: t.primary_key for t in schema.tables}
    table_map = schema.get_table_map()

    fks: List[Tuple[str, str, str, str]] = []
    for fk in schema.relationships or []:
        referred = table_map.get(fk.referred_table)
        if referred is None or not referred.primary_key:
            logger.warning(
                f"[Stage 3] auto-sharding: skipping FK {fk.referencing_table}."
                f"{fk.referencing_column} -> {fk.referred_table}: referred "
                f"table has no single-column PK."
            )
            continue
        fks.append(
            (
                fk.referencing_table,
                fk.referencing_column,
                fk.referred_table,
                referred.primary_key[0],
            )
        )

    fact_col_map = build_fact_column_map(schema, facts)
    ilp_facts: Dict[str, List[Tuple[str, str]]] = {}
    for fact in facts:
        cols = fact_col_map.get(fact.id, [])
        distinct_tables = {t for t, _ in cols}
        if len(distinct_tables) > _MAX_MENTIONED_TABLES_FOR_SHARDING:
            ilp_facts[str(fact.id)] = []
            continue
        ilp_facts[str(fact.id)] = cols

    return tables, columns_by_table, pks_by_table, fks, ilp_facts


def _shard_dicts_to_schemas(
    shard_dicts: List[Dict[str, List[str]]], global_schema: Schema
) -> List[Schema]:
    table_map = global_schema.get_table_map()
    shard_schemas: List[Schema] = []
    for shard_tables in shard_dicts:
        tables: List[Table] = []
        for table_name, col_names in shard_tables.items():
            src_table = table_map[table_name]
            col_set = set(col_names)
            tables.append(
                Table(
                    name=src_table.name,
                    columns=[c for c in src_table.columns if c.name in col_set],
                    primary_key=src_table.primary_key,
                )
            )
        relationships = [
            fk
            for fk in (global_schema.relationships or [])
            if fk.referencing_table in shard_tables
            and fk.referred_table in shard_tables
            and fk.referencing_column in shard_tables[fk.referencing_table]
        ]
        shard_schemas.append(Schema(tables=tables, relationships=relationships))
    return shard_schemas


def _derive_shards_from_schema(
    schema: Schema, facts: List[AtomicFact], model: Optional[str]
) -> List[Schema]:
    """The no-manual-hyperparameters entry point orchestrate() falls back to
    when callers don't supply pre-sharded `shards` themselves: derives
    max_shards/max_tables_per_shard/ILP weights automatically from the
    schema, the fact set, and the target provider/model's real context
    window (via shard_schema_auto()), never exposing those knobs here."""
    provider, api_key, _base_url, default_model = _detect_provider()
    resolved_model = model or default_model
    tables, columns_by_table, pks_by_table, fks, ilp_facts = _build_ilp_inputs(
        schema, facts
    )
    shard_dicts, _shard_facts = shard_schema_auto(
        tables,
        columns_by_table,
        pks_by_table,
        fks,
        ilp_facts,
        provider=provider,
        model=resolved_model,
        api_key=api_key,
    )
    if shard_dicts is None:
        raise RuntimeError(
            "[Stage 3] automatic sharding found no feasible solution for this "
            "schema/fact set."
        )
    logger.info(
        f"[Stage 3] Automatic sharding produced {len(shard_dicts)} shard(s): "
        + ", ".join(f"{list(st.keys())}" for st in shard_dicts)
    )
    return _shard_dicts_to_schemas(shard_dicts, schema)
