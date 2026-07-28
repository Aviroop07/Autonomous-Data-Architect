"""Maps each fact to the exact (table, column) pairs it touches -- the
input the ILP sharder (sharding_ilp.py) needs to decide which columns
must co-locate, and which facts genuinely drive cross-table cohesion.

Replaces the old crude approach (still visible in git history as
stage3/entry.py's _build_ilp_inputs): detect mentioned TABLES by name via
find_mentioned_tables(), then grab EVERY column of every mentioned table.
That over-broad mapping is what caused the ILP's hard fact-containment
constraint to force whole-table duplication even for facts that only
express a narrow cardinality relationship (e.g. "each order contains 1-15
order items" only needs order_id on both sides, not all 9 of ORDER's
columns) -- confirmed via a real live run and unit-tested regression
(TestNoSpuriousDuplication in test_sharding_auto.py).

Three layered signals, in priority order:
  1. The schema's own per-column source_fact_ids -- the PRIMARY, most
     precise signal, reliable now that Stage 2's er_extractor/er_auditor
     prompts were fixed to attribute REFERENCE facts (a statistical or
     conditional fact that mentions an existing attribute without
     creating it), not just CREATION facts, to the specific attributes
     they touch.
  2. FK relationship source_fact_ids -- catches structural "this
     relationship exists" facts that Stage 2 sometimes attaches only to
     the relationship object, not any specific column.
  3. Fallback, for any fact NEITHER of the above captured: text-mention
     table detection (find_mentioned_tables) + FK-graph path connectivity
     between the mentioned tables -- a direct edge if two mentioned
     tables are FK-connected, or the shortest path through intermediate
     "hop" tables otherwise, including only the PK/FK columns actually
     needed to make the join representable. Facts with no path between
     their mentioned tables (disconnected FK components) are skipped for
     this purpose -- forcing co-location across genuinely unrelated parts
     of the schema isn't meaningful either way.

Facts with zero resolvable columns get an empty list, matching the
existing ILP convention (excluded from the hard fact-containment
constraint via HC7a's h=0 forcing, never dropped from the fact list
passed to orchestrate() itself).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.schema import Schema, Table
from src.pipeline.stage3.middleware.fact_allocation import find_mentioned_tables


def _build_fk_adjacency(schema: Schema) -> Dict[str, List[Tuple[str, str, str]]]:
    """Undirected adjacency over the FK graph: table -> list of
    (neighbor_table, own_column, neighbor_column) reachable via one FK
    hop in either direction. Only FKs whose referred table has a single-
    column PK are usable as a join edge (matching the ILP's own
    convention elsewhere for skipping composite-PK-referencing FKs)."""
    adj: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    table_map = schema.get_table_map()
    for fk in schema.relationships or []:
        ref_table = table_map.get(fk.referred_table)
        if ref_table is None or len(ref_table.primary_key) != 1:
            continue
        pk = ref_table.primary_key[0]
        adj[fk.referencing_table].append((fk.referred_table, fk.referencing_column, pk))
        adj[fk.referred_table].append((fk.referencing_table, pk, fk.referencing_column))
    return adj


def _shortest_path_edges(
    adj: Dict[str, List[Tuple[str, str, str]]], start: str, end: str
) -> List[Tuple[str, str, str, str]]:
    """BFS shortest path from start to end table via the FK adjacency.
    Returns the list of (table, column, neighbor_table, neighbor_column)
    edges traversed, or [] if start==end or no path exists."""
    if start == end:
        return []
    visited = {start}
    queue: deque = deque([(start, [])])
    while queue:
        node, path = queue.popleft()
        for neighbor, own_col, neighbor_col in adj.get(node, []):
            if neighbor in visited:
                continue
            new_path = path + [(node, own_col, neighbor, neighbor_col)]
            if neighbor == end:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))
    return []


def _connect_via_fk_paths(
    adj: Dict[str, List[Tuple[str, str, str]]], mentioned: Set[str]
) -> Set[Tuple[str, str]]:
    """For a fact mentioning 2+ tables, connects every pair of them via
    the FK graph (union of pairwise shortest paths, including any
    intermediate 'hop' tables not originally mentioned) and returns the
    set of (table, column) PK/FK pairs along the paths actually used."""
    result: Set[Tuple[str, str]] = set()
    mentioned_list = sorted(mentioned)
    for i in range(len(mentioned_list)):
        for j in range(i + 1, len(mentioned_list)):
            for t1, c1, t2, c2 in _shortest_path_edges(
                adj, mentioned_list[i], mentioned_list[j]
            ):
                result.add((t1, c1))
                result.add((t2, c2))
    return result


def build_fact_column_map(
    schema: Schema, facts: List[AtomicFact]
) -> Dict[int, List[Tuple[str, str]]]:
    """Returns fact_id -> sorted list of (table, column) pairs that fact
    genuinely touches, per the three layered signals described above."""
    table_map: Dict[str, Table] = schema.get_table_map()
    table_names = [t.name for t in schema.tables]

    fact_to_cols: Dict[int, Set[Tuple[str, str]]] = defaultdict(set)

    # Signal 1: per-column provenance (primary, most precise)
    for table in schema.tables:
        for col in table.columns:
            for fid in col.source_fact_ids:
                fact_to_cols[fid].add((table.name, col.name))

    # Table-level-only attribution (whole-entity/behavioral facts with no
    # specific column pinned, e.g. a state-lifecycle rule) -- anchor on
    # the table's PK so the ILP can still resolve it structurally.
    for table in schema.tables:
        col_level_fids = {fid for c in table.columns for fid in c.source_fact_ids}
        table_only_fids = set(table.source_fact_ids) - col_level_fids
        for fid in table_only_fids:
            for pk in table.primary_key:
                fact_to_cols[fid].add((table.name, pk))

    # Signal 2: FK relationship provenance
    for fk in schema.relationships or []:
        ref_table = table_map.get(fk.referred_table)
        if ref_table is None or not ref_table.primary_key:
            continue
        for fid in fk.source_fact_ids:
            fact_to_cols[fid].add((fk.referencing_table, fk.referencing_column))
            fact_to_cols[fid].add((fk.referred_table, ref_table.primary_key[0]))

    # Signal 3: fallback for anything neither signal above captured
    fk_graph = _build_fk_adjacency(schema)
    for fact in facts:
        if fact_to_cols.get(fact.id):
            continue
        mentioned = find_mentioned_tables(fact.fact, table_names)
        if not mentioned:
            continue
        if len(mentioned) == 1:
            t = next(iter(mentioned))
            for pk in table_map[t].primary_key:
                fact_to_cols[fact.id].add((t, pk))
            continue
        fact_to_cols[fact.id].update(_connect_via_fk_paths(fk_graph, mentioned))

    return {fid: sorted(cols) for fid, cols in fact_to_cols.items()}
