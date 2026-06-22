from typing import List, Dict, NamedTuple, Set, Tuple, Optional
from src.pipeline.stage2.models.schema import Schema, Table, ForeignKey
from src.util.algorithms.matching import gale_shapley_matching
from src.pipeline.stage2.middleware.schema_merging.similarity import (
    get_similarity_score,
    get_similarity_matrix,
)

from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.models.merge_decision import (
    MergeDecisionLog,
    TableMatchDecision,
    UnmatchedTable,
)


class _JunctionEntry(NamedTuple):
    table: str
    entities: Set[str]


class _TableScoreBreakdown(NamedTuple):
    total: float
    name_score: float  # pre-modifier-penalty
    attr_score: float
    had_modifier_penalty: bool


class SchemaMerger:
    def __init__(
        self, alpha: float = 0.8, col_thresh: float = 0.8, table_thresh: float = 0.8
    ):
        self.alpha = alpha
        self.col_thresh = col_thresh
        self.table_thresh = table_thresh

    def merge_segments(
        self,
        segments: List[Schema],
        registry: Optional[TableFactRegistry] = None,
        strict: bool = False,
    ) -> Tuple[Schema, MergeDecisionLog]:
        """
        Merges multiple schema segments sequentially.
        Returns the merged schema and a log of all merge decisions made.

        strict=True: match tables by identical name only (for the final post-audit re-merge,
        where canonical names are already stable and fuzzy matching risks collapsing distinct
        entities with similar attribute sets).
        """
        log = MergeDecisionLog()

        if not segments:
            return Schema(tables=[]), log

        if len(segments) == 1:
            res = segments[0].model_copy(deep=True)
            res.normalize(registry=registry)
            res.wire_orphan_fk_columns()
            return res, log

        result = segments[0].model_copy(deep=True)
        result.normalize(registry=registry)

        for i in range(1, len(segments)):
            next_segment = segments[i].model_copy(deep=True)
            next_segment.normalize(registry=registry)
            result = self.merge_two_segments(
                result, next_segment, registry, log=log, strict=strict
            )

        # Post-Merge Enhancements
        # NOTE: _infer_cross_shard_fks removed — MergeReviewAgent + auditors handle cross-shard FK inference
        self._consolidate_junction_relationships(result)
        self._repair_relationship_table_names(result)
        result.align_fk_column_types()
        result.wire_orphan_fk_columns()

        return result, log

    def merge_two_segments(
        self,
        A: Schema,
        B: Schema,
        registry: Optional[TableFactRegistry] = None,
        log: Optional[MergeDecisionLog] = None,
        strict: bool = False,
    ) -> Schema:
        """
        Merges two schema segments A and B. Updates A and returns it.

        strict=True: skip GS scoring and match tables by identical name only.
        """

        # A and B are assumed to be normalized already

        table_list_a = A.tables
        table_list_b = B.tables

        if not table_list_a:
            return B.model_copy(deep=True)
        if not table_list_b:
            return A

        score_matrix: List[List[float]] = []
        score_details: Dict[Tuple[int, int], _TableScoreBreakdown] = {}

        if strict:
            # Name-exact matching: only tables with identical names (case-insensitive) merge.
            # This is the correct semantic for the final post-audit re-merge, where canonical
            # table names are stable and fuzzy scoring risks collapsing distinct entities.
            name_to_idx_a = {t.name.upper(): i for i, t in enumerate(table_list_a)}
            table_matches: List[Tuple[int, int]] = []
            for j, t2 in enumerate(table_list_b):
                if t2.name.upper() in name_to_idx_a:
                    idx_a = name_to_idx_a[t2.name.upper()]
                    table_matches.append((idx_a, j))
                    score_details[(idx_a, j)] = _TableScoreBreakdown(
                        total=1.0,
                        name_score=1.0,
                        attr_score=0.0,
                        had_modifier_penalty=False,
                    )
            print(f"  [Strict] Name-exact matching: {len(table_matches)} match(es)")
            for idx_a, idx_b in table_matches:
                print(
                    f"    MATCH  {table_list_a[idx_a].name:<32} <- {table_list_b[idx_b].name:<32}  (exact name)"
                )
        else:
            # 1. Calculate table score matrix
            for i, t1 in enumerate(table_list_a):
                row = []
                for j, t2 in enumerate(table_list_b):
                    breakdown = self._calculate_table_score(t1, t2)
                    row.append(breakdown.total)
                    score_details[(i, j)] = breakdown
                score_matrix.append(row)

            # 2. Match tables via Gale-Shapley
            print(
                f"  [GS] Score matrix ({len(table_list_a)} A x {len(table_list_b)} B), thresh={self.table_thresh}:"
            )
            for i, t1 in enumerate(table_list_a):
                for j, t2 in enumerate(table_list_b):
                    bd = score_details[(i, j)]
                    print(
                        f"    {t1.name:<32} vs {t2.name:<32} total={bd.total:.3f}  name={bd.name_score:.3f}  attr={bd.attr_score:.3f}"
                    )
            table_matches = gale_shapley_matching(score_matrix, self.table_thresh)
            print(f"  [GS] Matched {len(table_matches)} pair(s):")
            for idx_a, idx_b in table_matches:
                bd = score_details[(idx_a, idx_b)]
                print(
                    f"    MATCH  {table_list_a[idx_a].name:<32} <- {table_list_b[idx_b].name:<32}  score={bd.total:.3f}"
                )
            unmatched_b_names = [
                table_list_b[j].name
                for j in range(len(table_list_b))
                if j not in {m[1] for m in table_matches}
            ]
            if unmatched_b_names:
                print(
                    f"  [GS] Unmatched B-tables (will be added or collision-merged): {unmatched_b_names}"
                )

        matched_indices_b = {m[1] for m in table_matches}

        # 3. Merge matched tables (Multi-pass validation)
        # Pass 1: Pre-calculate all column matches
        # [NEW] Pre-identify name-based "matches" to allow FK validation for name-colliding tables
        table_map_b_to_a = {
            table_list_b[idx_b].name: table_list_a[idx_a].name
            for idx_a, idx_b in table_matches
        }
        for t2 in table_list_b:
            if t2.name not in table_map_b_to_a:
                existing_a = next(
                    (t1 for t1 in table_list_a if t1.name == t2.name), None
                )
                if existing_a:
                    table_map_b_to_a[t2.name] = existing_a.name

        all_col_matches: Dict[
            Tuple[str, str], List[Tuple[int, int]]
        ] = {}  # (table_a, table_b) -> matches

        col_name_maps: Dict[Tuple[str, str], Dict[str, str]] = {}

        for idx_a, idx_b in table_matches:
            t1 = table_list_a[idx_a]
            t2 = table_list_b[idx_b]
            cols1 = [c.name for c in t1.columns]
            cols2 = [c.name for c in t2.columns]
            col_score_matrix = get_similarity_matrix(cols1, cols2)
            col_matches = gale_shapley_matching(col_score_matrix, self.col_thresh)

            # [HARDENING] Force match PKs for matched tables only when clearly aligned
            pk1 = t1.pk
            pk2 = t2.pk
            if pk1 and pk2:
                pk_idx_a = next(
                    (i for i, c in enumerate(t1.columns) if c.name == pk1), -1
                )
                pk_idx_b = next(
                    (i for i, c in enumerate(t2.columns) if c.name == pk2), -1
                )

                # If PKs are not already matched, force them
                if pk_idx_a != -1 and pk_idx_b != -1:
                    is_matched = any(
                        m[0] == pk_idx_a or m[1] == pk_idx_b for m in col_matches
                    )
                    if not is_matched and self._should_force_match_pks(pk1, pk2):
                        print(
                            f"    [Merger] Force-matching PKs: {t1.name}.{pk1} <-> {t2.name}.{pk2}"
                        )
                        col_matches.append((pk_idx_a, pk_idx_b))

            all_col_matches[(t1.name, t2.name)] = col_matches
            col_name_maps[(t1.name, t2.name)] = {
                cols2[idx_b]: cols1[idx_a] for idx_a, idx_b in col_matches
            }

        # Pass 2: Validate FK consistency for matched columns
        fk_divergent_pairs: Set[Tuple[str, str]] = set()
        for (name_a, name_b), col_matches in all_col_matches.items():
            t1 = next(t for t in table_list_a if t.name == name_a)
            t2 = next(t for t in table_list_b if t.name == name_b)

            for c_idx_a, c_idx_b in col_matches:
                c1 = t1.columns[c_idx_a]
                c2 = t2.columns[c_idx_b]

                # Check for FKs in A and B
                fk1 = next(
                    (
                        r
                        for r in (A.relationships or [])
                        if r.referencing_table == t1.name
                        and r.referencing_column == c1.name
                    ),
                    None,
                )
                fk2 = next(
                    (
                        r
                        for r in (B.relationships or [])
                        if r.referencing_table == t2.name
                        and r.referencing_column == c2.name
                    ),
                    None,
                )

                if fk1 and fk2:
                    # Both are FKs, verify targets
                    # Goal: target tables MUST be matched, and target columns MUST be matched
                    target_table_a = fk1.referred_table
                    target_table_b = fk2.referred_table

                    t_ref_a = next(t for t in table_list_a if t.name == target_table_a)
                    t_ref_b = next(t for t in table_list_b if t.name == target_table_b)

                    target_col_a = t_ref_a.pk
                    target_col_b = t_ref_b.pk

                    # 1. Target tables must be a matched pair (either via GS match or name collision)
                    if table_map_b_to_a.get(target_table_b) != target_table_a:
                        # Case 63 Fix: If names match, it's fine, the map should have it now.
                        # If it's still missing, it's a real mapping error.
                        print(
                            f"    [Merger Warning] FK Target Mismatch: {t1.name}.{c1.name} and {t2.name}.{c2.name} are matched, but target tables ({target_table_a} vs {target_table_b}) are not matched. Treating as independent unless name-merged later."
                        )
                        continue

                    # 2. Target columns within those matched tables must be a matched pair
                    target_col_matches = all_col_matches.get(
                        (target_table_a, target_table_b), []
                    )
                    # find if (target_col_idx_a, target_col_idx_b) is in target_col_matches

                    try:
                        t_idx_a = next(
                            i
                            for i, c in enumerate(t_ref_a.columns)
                            if c.name == target_col_a
                        )
                        t_idx_b = next(
                            i
                            for i, c in enumerate(t_ref_b.columns)
                            if c.name == target_col_b
                        )
                        if (t_idx_a, t_idx_b) not in target_col_matches:
                            # Instead of raising ValueError, we log and try to canonicalize if they are both PKs
                            if (
                                target_col_a == t_ref_a.pk
                                and target_col_b == t_ref_b.pk
                            ):
                                print(
                                    f"    [Merger Warning] FK Target Divergence: {t1.name}.{c1.name} and {t2.name}.{c2.name} point to un-matched PKs ({target_col_a} vs {target_col_b}). Normalizing to {target_col_a}."
                                )
                                fk_divergent_pairs.add((name_a, name_b))
                                # We don't raise error, the later merge will unify the names
                            else:
                                print(
                                    f"    [Merger Error] Irreconcilable FK Conflict: {t1.name}.{c1.name} and {t2.name}.{c2.name} match, but point to distinct non-PK columns ({target_col_a} vs {target_col_b})."
                                )
                    except StopIteration:
                        # This should not happen if schemas are consistent
                        pass

        # Pass 3: Actual Merge
        for idx_a, idx_b in table_matches:
            t1 = table_list_a[idx_a]
            t2 = table_list_b[idx_b]
            # Capture fact IDs before the registry merge combines them
            pre_fact_ids_a = (
                list(registry.get_facts_for_tables([t1.name])) if registry else []
            )
            pre_fact_ids_b = (
                list(registry.get_facts_for_tables([t2.name])) if registry else []
            )
            if registry:
                registry.merge_tables(t2.name, t1.name)
            col_matches = all_col_matches[(t1.name, t2.name)]
            self._merge_tables(A, t1, t2, col_matches, registry)
            if log is not None:
                breakdown = score_details[(idx_a, idx_b)]
                log.matched_pairs.append(
                    TableMatchDecision(
                        shard_a_table=t1.name,
                        shard_b_table=t2.name,
                        match_score=breakdown.total,
                        name_score=breakdown.name_score,
                        attr_score=breakdown.attr_score,
                        matched_columns=[t1.columns[ia].name for ia, _ in col_matches],
                        shard_a_fact_ids=pre_fact_ids_a,
                        shard_b_fact_ids=pre_fact_ids_b,
                        had_pk_divergence=bool(t1.pk)
                        and bool(t2.pk)
                        and t1.pk != t2.pk,
                        had_fk_target_divergence=(t1.name, t2.name)
                        in fk_divergent_pairs,
                    )
                )
                if breakdown.had_modifier_penalty:
                    log.modifier_penalty_applied.append(f"{t1.name}::{t2.name}")

        # 4. Add unmatched tables from B to A OR force merge on name collision
        # Capture before the loop: A.tables grows as unmatched B-tables are appended,
        # but score_matrix was built on the original a-table count.
        num_original_a_tables = len(table_list_a)
        for i, t2 in enumerate(table_list_b):
            if i not in matched_indices_b:
                # Collision Check
                existing_table = next((t for t in A.tables if t.name == t2.name), None)
                if log is not None and not strict and num_original_a_tables > 0:
                    best_idx_a = max(
                        range(num_original_a_tables), key=lambda j: score_matrix[j][i]
                    )
                    log.unmatched_tables.append(
                        UnmatchedTable(
                            shard_b_table=t2.name,
                            reason="name_collision_merged"
                            if existing_table
                            else "new_entity",
                            best_candidate_in_a=table_list_a[best_idx_a].name,
                            best_candidate_score=score_matrix[best_idx_a][i],
                            shard_b_fact_ids=list(
                                registry.get_facts_for_tables([t2.name])
                            )
                            if registry
                            else [],
                        )
                    )
                if existing_table:
                    # Name collision not captured by GS (below threshold or capacity
                    # exhausted). Run explicit column matching so same-concept columns
                    # are aligned rather than blindly appended.
                    cols1 = [c.name for c in existing_table.columns]
                    cols2 = [c.name for c in t2.columns]
                    col_score_matrix = get_similarity_matrix(cols1, cols2)
                    name_col_matches = gale_shapley_matching(
                        col_score_matrix, self.col_thresh
                    )
                    print(
                        f"    [Merger] Name-collision merge: {existing_table.name} "
                        f"({len(name_col_matches)} col matches)"
                    )
                    if registry:
                        registry.merge_tables(t2.name, existing_table.name)
                    self._merge_tables(
                        A, existing_table, t2, name_col_matches, registry
                    )
                else:
                    # Truly new table
                    new_table = t2.model_copy(deep=True)
                    A.tables.append(new_table)
                    print(f"    [GS] UNMATCHED {t2.name} added as new table")

        # 5. Merge relationships
        if B.relationships:
            if A.relationships is None:
                A.relationships = []

            for rel in B.relationships:
                new_rel = rel.model_copy(deep=True)
                # Remap tables
                original_referencing_table = new_rel.referencing_table
                new_rel.referencing_table = table_map_b_to_a.get(
                    new_rel.referencing_table, new_rel.referencing_table
                )
                new_rel.referred_table = table_map_b_to_a.get(
                    new_rel.referred_table, new_rel.referred_table
                )

                # Remap referencing columns when matched columns were merged
                col_map = col_name_maps.get(
                    (new_rel.referencing_table, original_referencing_table), {}
                )
                if new_rel.referencing_column in col_map:
                    new_rel.referencing_column = col_map[new_rel.referencing_column]

                # Check for uniqueness before adding
                exists = False
                for r in A.relationships:
                    if (
                        r.referencing_table == new_rel.referencing_table
                        and r.referencing_column == new_rel.referencing_column
                        and r.referred_table == new_rel.referred_table
                    ):
                        exists = True
                        break
                if not exists:
                    A.relationships.append(new_rel)

        return A

    @staticmethod
    def _tokenize_table_name(name: str) -> List[str]:
        return [token for token in name.upper().split("_") if token]

    def _has_distinct_modifiers(self, name_a: str, name_b: str) -> bool:
        tokens_a = self._tokenize_table_name(name_a)
        tokens_b = self._tokenize_table_name(name_b)
        if not tokens_a or not tokens_b:
            return False

        head_a = tokens_a[-1]
        head_b = tokens_b[-1]
        if head_a != head_b:
            return False

        modifiers_a = set(tokens_a[:-1])
        modifiers_b = set(tokens_b[:-1])
        if not modifiers_a and not modifiers_b:
            return False

        return modifiers_a != modifiers_b

    @staticmethod
    def _is_identifier_column(name: str) -> bool:
        lowered = name.lower()
        if lowered in {"id", "pk"}:
            return True
        return (
            lowered.endswith("_id")
            or lowered.endswith("_uuid")
            or lowered.endswith("_guid")
        )

    def _matches_id_only(
        self, cols1: List[str], cols2: List[str], col_matches: List[Tuple[int, int]]
    ) -> bool:
        if not col_matches:
            return False
        for idx_a, idx_b in col_matches:
            if not (
                self._is_identifier_column(cols1[idx_a])
                and self._is_identifier_column(cols2[idx_b])
            ):
                return False
        return True

    def _calculate_table_score(self, t1: Table, t2: Table) -> _TableScoreBreakdown:
        raw_name_score = get_similarity_score(t1.name, t2.name)

        # Exact Name Match Boost
        if t1.name.upper() == t2.name.upper():
            raw_name_score = 1.0

        had_modifier_penalty = False
        name_score = raw_name_score
        if name_score < 0.98 and self._has_distinct_modifiers(t1.name, t2.name):
            had_modifier_penalty = True
            name_score *= 0.3

        # Column matching
        cols1 = [c.name for c in t1.columns]
        cols2 = [c.name for c in t2.columns]

        col_score_matrix = get_similarity_matrix(cols1, cols2)
        col_matches = gale_shapley_matching(col_score_matrix, self.col_thresh)

        # Normalize attr_score by the number of columns in the smaller table (or max)
        if not col_matches:
            attr_score = 0.0
        else:
            total_match_similarity = sum(
                col_score_matrix[m[0]][m[1]] for m in col_matches
            )
            possible_columns = max(len(cols1), len(cols2))
            attr_score = total_match_similarity / possible_columns

            if name_score < 0.9 and self._matches_id_only(cols1, cols2, col_matches):
                attr_score = 0.0

        return _TableScoreBreakdown(
            total=self.alpha * name_score + (1.0 - self.alpha) * attr_score,
            name_score=raw_name_score,
            attr_score=attr_score,
            had_modifier_penalty=had_modifier_penalty,
        )

    def _should_force_match_pks(self, pk1: str, pk2: str) -> bool:
        if self._is_identifier_column(pk1) and self._is_identifier_column(pk2):
            return True

        return get_similarity_score(pk1, pk2) >= 0.85

    def _merge_tables(
        self,
        schema_a: Schema,
        t1: Table,
        t2: Table,
        col_matches: List[Tuple[int, int]],
        registry: Optional[TableFactRegistry] = None,
    ):
        """
        Merges t2 into t1 within schema_a using pre-calculated column matches.
        Uses Union-based merging.
        """
        matched_indices_b = {m[1] for m in col_matches}

        # Add unmatched columns from t2 to t1 (Information Maximization)
        existing_col_names = {c.name.lower() for c in t1.columns}
        for i, c2 in enumerate(t2.columns):
            if i not in matched_indices_b:
                if c2.name.lower() not in existing_col_names:
                    t1.columns.append(c2.model_copy(deep=True))
                    existing_col_names.add(c2.name.lower())

        # Merge unique constraints
        if t2.unique:
            if t1.unique is None:
                t1.unique = []
            for uq2 in t2.unique:
                exists = False
                for uq1 in t1.unique:
                    if set(uq1.columns) == set(uq2.columns):
                        exists = True
                        break
                if not exists:
                    t1.unique.append(uq2.model_copy(deep=True))

    def _infer_cross_shard_fks(self, schema: Schema):
        """
        Injects foreign keys based on naming conventions ({table}_id)
        when both entities exist in the merged schema but lacked a relationship.
        """
        table_names = {t.name.upper(): t.name for t in schema.tables}
        if schema.relationships is None:
            schema.relationships = []

        existing_rels = {
            (
                r.referencing_table.upper(),
                r.referencing_column.upper(),
                r.referred_table.upper(),
            )
            for r in schema.relationships
        }

        for table in schema.tables:
            for col in table.columns:
                if col.name == table.pk:
                    continue
                col_name = col.name.lower()
                if col_name.endswith("_id"):
                    target_table_name_upper = col_name[:-3].upper()
                    if (
                        target_table_name_upper in table_names
                        and target_table_name_upper != table.name.upper()
                    ):
                        # Potential FK
                        rel_key = (
                            table.name.upper(),
                            col.name.upper(),
                            target_table_name_upper,
                        )
                        if rel_key not in existing_rels:
                            # Add the FK
                            new_fk = ForeignKey(
                                referencing_table=table.name,
                                referencing_column=col.name,
                                referred_table=table_names[target_table_name_upper],
                            )
                            schema.relationships.append(new_fk)
                            existing_rels.add(rel_key)

    def validate_connectivity(self, schema: Schema) -> List[str]:
        """
        Detects orphaned tables that are not reachable via direct FKs.
        """
        if not schema.tables:
            return []

        # Build adjacency graph
        adj = {t.name.upper(): set() for t in schema.tables}
        if schema.relationships:
            for rel in schema.relationships:
                t1 = rel.referencing_table.upper()
                t2 = rel.referred_table.upper()
                if t1 in adj and t2 in adj:
                    adj[t1].add(t2)
                    adj[t2].add(t1)

        # Find connected components (simple BFS/DFS)
        nodes = list(adj.keys())
        visited = set()
        components = 0

        for node in nodes:
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        stack.extend(adj[curr] - visited)

        findings = []
        if components > 1:
            findings.append(
                f"Schema is fragmented into {components} disconnected components."
            )
            # Identify isolated tables (components of size 1)
            for node, neighbors in adj.items():
                if not neighbors:
                    findings.append(f"Table '{node}' is strictly isolated.")
        return findings

    def _consolidate_junction_relationships(self, schema: Schema):
        """
        [HARDENING] Identifies and removes redundant direct FKs when a
        Many-to-Many junction table exists for the same relationship.
        """
        if not schema.relationships or not schema.tables:
            return

        junction_tables = []
        for table in schema.tables:
            # A junction table should be a minimal join: PK + 2 FK columns only.
            fks = [r for r in schema.relationships if r.referencing_table == table.name]
            if len(fks) != 2:
                continue

            entity_a = fks[0].referred_table
            entity_b = fks[1].referred_table
            if entity_a == entity_b:
                continue

            fk_cols = {r.referencing_column for r in fks}
            non_pk_cols = {c.name for c in table.columns if c.name != table.pk}

            # If the table has extra non-FK columns, it is not a pure junction.
            if not non_pk_cols or not non_pk_cols.issubset(fk_cols):
                continue

            junction_tables.append(
                {"table": table.name, "entities": {entity_a.upper(), entity_b.upper()}}
            )

        # Find and remove redundant direct FKs
        to_remove = []
        for i, rel in enumerate(schema.relationships):
            # If this is a direct FK between two entities that also have a junction table
            t1 = rel.referencing_table.upper()
            t2 = rel.referred_table.upper()

            for junc in junction_tables:
                if junc["entities"] == {t1, t2}:
                    # Conflict! Direct FK exists where a junction exists.
                    print(
                        f"    [Merger] Removing redundant direct FK: {rel.referencing_table}.{rel.referencing_column} -> {rel.referred_table} (Junction {junc['table']} takes precedence)"
                    )
                    to_remove.append(i)
                    break

        # Perform removal in reverse to maintain indices
        for idx in sorted(to_remove, reverse=True):
            schema.relationships.pop(idx)

    def _repair_relationship_table_names(self, schema: Schema) -> None:
        if not schema.relationships or not schema.tables:
            return

        table_names = [t.name for t in schema.tables]
        table_names_upper = {t.name.upper() for t in schema.tables}

        def resolve(name: str) -> Optional[str]:
            if name.upper() in table_names_upper:
                return name
            scores = [
                (candidate, get_similarity_score(name, candidate))
                for candidate in table_names
            ]
            scores.sort(key=lambda pair: pair[1], reverse=True)
            if not scores:
                return None
            best_name, best_score = scores[0]
            second_score = scores[1][1] if len(scores) > 1 else 0.0
            if best_score >= 0.95 and (best_score - second_score) >= 0.05:
                return best_name
            return None

        for rel in schema.relationships:
            new_ref_table = resolve(rel.referencing_table)
            if new_ref_table:
                rel.referencing_table = new_ref_table

            new_target_table = resolve(rel.referred_table)
            if new_target_table:
                rel.referred_table = new_target_table
