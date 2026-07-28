import sys
import os
import numpy as np
from typing import Any, Callable, List, Sequence, Set, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util as st_util
import nltk
from nltk.corpus import wordnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.util.schema_model.schema import Schema, Table


COARSE_DT_MAP: Dict[str, str] = {
    "INT": "NUMERIC",
    "INTEGER": "NUMERIC",
    "BIGINT": "NUMERIC",
    "SMALLINT": "NUMERIC",
    "TINYINT": "NUMERIC",
    "FLOAT": "NUMERIC",
    "DOUBLE": "NUMERIC",
    "REAL": "NUMERIC",
    "DECIMAL": "NUMERIC",
    "NUMERIC": "NUMERIC",
    "NUMBER": "NUMERIC",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "TEXT": "TEXT",
    "STRING": "TEXT",
    "NVARCHAR": "TEXT",
    "DATE": "DATETIME",
    "DATETIME": "DATETIME",
    "TIMESTAMP": "DATETIME",
    "TIME": "DATETIME",
    "BLOB": "BINARY",
    "BINARY": "BINARY",
    "BYTEA": "BINARY",
    "VARBINARY": "BINARY",
    "BOOLEAN": "BOOL",
    "BOOL": "BOOL",
    "BIT": "BOOL",
}


def coarsen_dt(dt: Optional[str]) -> str:
    if not dt:
        return "TEXT"
    base = str(dt).upper().split("(")[0].strip()
    return COARSE_DT_MAP.get(base, "TEXT")


def max_bipartite_matching(
    pred: Sequence[str],
    gt: Sequence[str],
    match_func: Callable[[str, str], bool],
) -> Dict[str, str]:
    """Maximum-cardinality matching between predicted and ground-truth names,
    returned as {gt_name: pred_name}.

    Replaces a greedy first-match scan, which was wrong in two ways that both
    mattered. It was ORDER-DEPENDENT: greedy strands a prediction whenever an
    earlier one consumes the only ground-truth name it could have matched, and
    since the callers passed Python `set`s -- whose iteration order is
    hash-randomised per process -- the resulting F1 changed between runs on
    identical input. And it UNDERCOUNTED: even with a fixed order, greedy
    finds a maximal matching, not a maximum one.

    Fuzzy name matching is genuinely many-to-many (short column names sit well
    above any reasonable cosine threshold: `interest_rate` vs `loan_amount`,
    `tier_id` vs `tier_name`), so those blocking choices are common rather than
    pathological.

    Kuhn's augmenting-path algorithm with an EXPLICIT STACK -- the textbook
    recursive form would hit Python's recursion limit on a schema with a few
    thousand columns, which is inside the range this project targets. Inputs
    are sorted so the result never depends on caller iteration order.
    """
    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    adjacency: Dict[str, List[str]] = {
        p: [g for g in gt_sorted if match_func(p, g)] for p in pred_sorted
    }

    # Both directions are maintained so flipping an augmenting path is O(path)
    # rather than a linear scan of the match map per step.
    matched_gt: Dict[str, str] = {}
    matched_pred: Dict[str, str] = {}

    for root in pred_sorted:
        # Iterative DFS for an augmenting path. `parent` records, for each gt
        # node reached, the pred node that reached it, so the alternating path
        # can be walked back once a free gt node is found.
        seen: Set[str] = set()
        parent: Dict[str, str] = {}
        stack: List[Tuple[str, int]] = [(root, 0)]
        found: Optional[str] = None
        while stack and found is None:
            p, idx = stack[-1]
            if idx >= len(adjacency[p]):
                stack.pop()
                continue
            stack[-1] = (p, idx + 1)
            g = adjacency[p][idx]
            if g in seen:
                continue
            seen.add(g)
            parent[g] = p
            if g in matched_gt:
                stack.append((matched_gt[g], 0))
            else:
                found = g
        if found is None:
            continue
        # Walk the alternating path back to the root, flipping each edge. The
        # root is unmatched by construction, so matched_pred.get(root) is None
        # and the walk terminates there.
        cursor: Optional[str] = found
        while cursor is not None:
            p = parent[cursor]
            previous = matched_pred.get(p)
            matched_gt[cursor] = p
            matched_pred[p] = cursor
            cursor = previous
    return matched_gt


class SchemaEvaluator:
    def __init__(self, sim_threshold: float = 0.6, lcs_threshold: float = 0.75) -> None:
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sim_threshold = sim_threshold
        self.lcs_threshold = lcs_threshold
        # Every name is compared against every other name, so without caching a
        # single schema costs O(names^2) transformer forward passes. Embeddings
        # are cached per name and match verdicts per ordered pair; both are pure
        # functions of the inputs, so caching cannot change a result.
        self._embedding_cache: Dict[str, Any] = {}
        self._match_cache: Dict[Tuple[str, str], bool] = {}
        self._synonym_cache: Dict[str, Set[str]] = {}
        try:
            wordnet.ensure_loaded()
        except Exception:
            nltk.download("wordnet")

    def prewarm(self, names: Sequence[str]) -> None:
        """Embed every name in one batched forward pass.

        Encoding one string at a time wastes almost all of the GPU/CPU batch
        width. Callers that know the full name vocabulary up front (i.e.
        evaluate_schema) should call this first; correctness does not depend on
        it, since _embed falls back to encoding on demand.
        """
        missing = sorted({n for n in names if n not in self._embedding_cache})
        if not missing:
            return
        vectors = self.sim_model.encode(missing, convert_to_tensor=True)
        for name, vector in zip(missing, vectors):
            self._embedding_cache[name] = vector

    def _embed(self, name: str):
        cached = self._embedding_cache.get(name)
        if cached is None:
            cached = self.sim_model.encode(name, convert_to_tensor=True)
            self._embedding_cache[name] = cached
        return cached

    def get_lcs_length(self, s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = 0
        return max_len

    def _synonyms(self, word: str) -> Set[str]:
        cached = self._synonym_cache.get(word)
        if cached is None:
            cached = set()
            for syn in wordnet.synsets(word):
                if syn is None:
                    continue
                for lemma in syn.lemmas():  # type: ignore[union-attr]
                    cached.add(lemma.name().lower())
            self._synonym_cache[word] = cached
        return cached

    def is_synonym(self, pred: str, gt: str) -> bool:
        return gt.lower() in self._synonyms(pred.lower())

    def get_similarity(self, s1: str, s2: str) -> float:
        return float(st_util.cos_sim(self._embed(s1), self._embed(s2))[0][0])

    def match_names(self, pred: str, gt: str) -> bool:
        key = (pred, gt)
        cached = self._match_cache.get(key)
        if cached is not None:
            return cached
        result = self._match_names_uncached(pred, gt)
        self._match_cache[key] = result
        return result

    def _match_names_uncached(self, pred: str, gt: str) -> bool:
        if pred.lower() == gt.lower():
            return True
        if self.is_synonym(pred.lower(), gt.lower()):
            return True
        if self.get_similarity(pred, gt) >= self.sim_threshold:
            return True
        lcs_len = self.get_lcs_length(pred.lower(), gt.lower())
        denom = max(len(pred), len(gt))
        if denom > 0 and lcs_len / denom >= self.lcs_threshold:
            return True
        return False

    def calculate_f1(
        self,
        pred_set: Set[str],
        gt_set: Set[str],
        match_func: Any,
    ) -> Tuple[float, float]:
        """F1 plus an exact-match indicator, over a MAXIMUM bipartite matching.

        `acc` is deliberately all-or-nothing: it is the standard exact-match
        companion to F1, i.e. "was this schema recovered perfectly", and is 1.0
        only when precision and recall are both 1.0. It is a function of f1, not
        independent evidence. Compared with a tolerance rather than `== 1.0`
        because f1 is computed, not exact.
        """
        if not gt_set and not pred_set:
            return 1.0, 1.0
        if not gt_set or not pred_set:
            return 0.0, 0.0
        matching = max_bipartite_matching(list(pred_set), list(gt_set), match_func)
        intersection_size = len(matching)
        precision = intersection_size / len(pred_set)
        recall = intersection_size / len(gt_set)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        acc = 1.0 if f1 >= 1.0 - 1e-9 else 0.0
        return f1, acc

    def evaluate_schema(
        self,
        pred_schema: Schema,
        gt_schema: Schema,
        gt_col_types: Optional[Dict[str, str]] = None,
        pred_col_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # Embed the whole name vocabulary in one batched pass before any
        # comparison happens -- see prewarm(). Purely a speed measure.
        self.prewarm(
            [t.name for t in gt_schema.tables]
            + [t.name for t in pred_schema.tables]
            + [c.name for t in gt_schema.tables for c in t.columns]
            + [c.name for t in pred_schema.tables for c in t.columns]
        )

        # 1. Table F1
        gt_table_names = {t.name for t in gt_schema.tables}
        pred_table_names = {t.name for t in pred_schema.tables}
        table_f1, table_acc = self.calculate_f1(
            pred_table_names, gt_table_names, self.match_names
        )

        # 2. Align tables -- maximum bipartite matching, for the same reason
        #    calculate_f1 uses it: a greedy scan both undercounts and depends on
        #    iteration order. Table alignment feeds every metric below it, so an
        #    unstable alignment would make PK/FK/DT unstable too.
        gt_by_name = {t.name: t for t in gt_schema.tables}
        pred_by_name = {t.name: t for t in pred_schema.tables}
        table_matching = max_bipartite_matching(
            list(pred_by_name), list(gt_by_name), self.match_names
        )
        # gt table name -> predicted Table. Every lookup below goes through this
        # map; the previous code re-scanned the match list per table, which is
        # O(T^2) on a schema with hundreds of tables.
        pred_for_gt: Dict[str, Table] = {
            gt_name: pred_by_name[pred_name]
            for gt_name, pred_name in table_matching.items()
        }

        # 3. Attribute F1 (averaged over GT tables)
        attr_f1_scores: List[float] = []
        for gt_t in gt_schema.tables:
            pred_t = pred_for_gt.get(gt_t.name)
            if pred_t:
                gt_attrs = {c.name for c in gt_t.columns}
                pred_attrs = {c.name for c in pred_t.columns}
                f1_attr, _ = self.calculate_f1(pred_attrs, gt_attrs, self.match_names)
                attr_f1_scores.append(f1_attr)
            else:
                attr_f1_scores.append(0.0)
        avg_attr_f1 = float(np.mean(attr_f1_scores)) if attr_f1_scores else 0.0
        attr_acc = 1.0 if avg_attr_f1 >= 1.0 - 1e-9 else 0.0

        # Reverse of pred_for_gt, for resolving a predicted FK's referred table
        # back into ground-truth naming.
        gt_name_for_pred_name: Dict[str, str] = {
            pred_t.name: gt_name for gt_name, pred_t in pred_for_gt.items()
        }

        # 4. PK accuracy -- exact match per Text2Schema protocol
        pk_correct = 0
        for gt_t in gt_schema.tables:
            pred_t = pred_for_gt.get(gt_t.name)
            if pred_t:
                gt_pk_set = {k.lower().strip() for k in gt_t.primary_key}
                pred_pk_set = {k.lower().strip() for k in pred_t.primary_key}
                if gt_pk_set == pred_pk_set:
                    pk_correct += 1
        pk_acc = pk_correct / len(gt_schema.tables) if gt_schema.tables else 1.0

        # 5. FK accuracy -- exact set match per Text2Schema protocol
        def get_fk_set(schema_obj: Schema, table_obj: Table) -> Set[Tuple[str, str]]:
            fks: Set[Tuple[str, str]] = set()
            if schema_obj.relationships:
                for rel in schema_obj.relationships:
                    if rel.referencing_table == table_obj.name:
                        fks.add((rel.referencing_column, rel.referred_table))
            return fks

        # Column alignment per matched table, computed once and reused by both
        # the FK and DT steps. Maximum matching again, rather than "first
        # predicted column that matches", which could bind one predicted column
        # to several ground-truth columns.
        col_alignment: Dict[str, Dict[str, str]] = {}
        for gt_t in gt_schema.tables:
            pred_t = pred_for_gt.get(gt_t.name)
            if pred_t is None:
                continue
            col_alignment[gt_t.name] = max_bipartite_matching(
                [c.name for c in pred_t.columns],
                [c.name for c in gt_t.columns],
                self.match_names,
            )
        pred_col_for_gt_col: Dict[str, Dict[str, str]] = {
            gt_table: {gt_col: pred_col for gt_col, pred_col in alignment.items()}
            for gt_table, alignment in col_alignment.items()
        }
        gt_col_for_pred_col: Dict[str, Dict[str, str]] = {
            gt_table: {pred_col: gt_col for gt_col, pred_col in alignment.items()}
            for gt_table, alignment in col_alignment.items()
        }

        fk_correct = 0
        for gt_t in gt_schema.tables:
            pred_t = pred_for_gt.get(gt_t.name)
            if pred_t:
                gt_fks = get_fk_set(gt_schema, gt_t)
                pred_fks_raw = get_fk_set(pred_schema, pred_t)
                local_cols = gt_col_for_pred_col.get(gt_t.name, {})
                pred_fks_mapped: Set[Tuple[str, str]] = {
                    (
                        local_cols.get(p_col, p_col),
                        gt_name_for_pred_name.get(p_ref_table, p_ref_table),
                    )
                    for p_col, p_ref_table in pred_fks_raw
                }
                if gt_fks == pred_fks_mapped:
                    fk_correct += 1
        fk_acc = fk_correct / len(gt_schema.tables) if gt_schema.tables else 1.0

        # 6. DT accuracy -- coarse 5-category, requires explicit type maps.
        #
        # Two figures, because one number was conflating two different failures.
        # `dt_acc` keeps the original denominator (EVERY ground-truth column),
        # so a column the pipeline never produced scores identically to a column
        # whose type it got wrong. `dt_acc_matched` divides only by the columns
        # that were actually aligned, which is the type-inference accuracy
        # proper; read together with attr F1 they separate recall from typing.
        dt_acc: Optional[float]
        dt_acc_matched: Optional[float]
        if gt_col_types is not None and pred_col_types is not None:
            dt_correct = 0
            dt_total = 0
            dt_matched_total = 0
            for gt_t in gt_schema.tables:
                pred_t = pred_for_gt.get(gt_t.name)
                aligned = pred_col_for_gt_col.get(gt_t.name, {})
                for gt_col in gt_t.columns:
                    dt_total += 1
                    if pred_t is None:
                        continue
                    matched_pred_col = aligned.get(gt_col.name)
                    if matched_pred_col is None:
                        continue
                    dt_matched_total += 1
                    gt_dt = coarsen_dt(gt_col_types.get(f"{gt_t.name}.{gt_col.name}"))
                    pred_dt = coarsen_dt(
                        pred_col_types.get(f"{pred_t.name}.{matched_pred_col}")
                    )
                    if gt_dt == pred_dt:
                        dt_correct += 1
            dt_acc = dt_correct / dt_total if dt_total > 0 else 1.0
            dt_acc_matched = (
                dt_correct / dt_matched_total if dt_matched_total > 0 else 1.0
            )
        else:
            dt_acc = None
            dt_acc_matched = None

        # 7. Attribute Coverage F1 (flat bag -- ignores table assignment)
        gt_attrs_flat = {c.name for t in gt_schema.tables for c in t.columns}
        pred_attrs_flat = {c.name for t in pred_schema.tables for c in t.columns}
        attr_coverage_f1, _ = self.calculate_f1(
            pred_attrs_flat, gt_attrs_flat, self.match_names
        )

        # 8. FD Coverage
        # 8a. PK FD Coverage: fuzzy PK match over matched tables
        pk_fd_match = 0
        for gt_t in gt_schema.tables:
            pred_t = pred_for_gt.get(gt_t.name)
            if pred_t:
                gt_pk_set = set(gt_t.primary_key)
                pred_pk_set = set(pred_t.primary_key)
                pk_f1, _ = self.calculate_f1(pred_pk_set, gt_pk_set, self.match_names)
                if pk_f1 >= 1.0 - 1e-9:
                    pk_fd_match += 1
        pk_fd_coverage = (
            pk_fd_match / len(gt_schema.tables) if gt_schema.tables else 1.0
        )

        # 8b. FK FD Coverage: global F1 over all FK triples with fuzzy matching
        gt_fk_triples = [
            (r.referencing_table, r.referencing_column, r.referred_table)
            for r in (gt_schema.relationships or [])
        ]
        pred_fk_triples = [
            (r.referencing_table, r.referencing_column, r.referred_table)
            for r in (pred_schema.relationships or [])
        ]
        if not gt_fk_triples and not pred_fk_triples:
            fk_fd_coverage = 1.0
        elif not gt_fk_triples or not pred_fk_triples:
            fk_fd_coverage = 0.0
        else:

            def _fk_matches(
                p_fk: Tuple[str, str, str], g_fk: Tuple[str, str, str]
            ) -> bool:
                return (
                    self.match_names(p_fk[0], g_fk[0])
                    and self.match_names(p_fk[1], g_fk[1])
                    and self.match_names(p_fk[2], g_fk[2])
                )

            matched_gt_fk_idx: Set[int] = set()
            matched_pred_fk_idx: Set[int] = set()
            for pi, pfk in enumerate(pred_fk_triples):
                for gi, gfk in enumerate(gt_fk_triples):
                    if gi not in matched_gt_fk_idx and _fk_matches(pfk, gfk):
                        matched_gt_fk_idx.add(gi)
                        matched_pred_fk_idx.add(pi)
                        break
            intersection = len(matched_gt_fk_idx)
            fk_precision = intersection / len(pred_fk_triples)
            fk_recall = intersection / len(gt_fk_triples)
            fk_fd_coverage = (
                2 * fk_precision * fk_recall / (fk_precision + fk_recall)
                if (fk_precision + fk_recall) > 0
                else 0.0
            )

        fd_coverage = (pk_fd_coverage + fk_fd_coverage) / 2.0

        return {
            "table_f1": table_f1,
            "table_acc": table_acc,
            "attr_f1": avg_attr_f1,
            "attr_acc": attr_acc,
            "pk_acc": pk_acc,
            "fk_acc": fk_acc,
            "dt_acc": dt_acc,
            "dt_acc_matched": dt_acc_matched,
            "attr_coverage_f1": attr_coverage_f1,
            "pk_fd_coverage": pk_fd_coverage,
            "fk_fd_coverage": fk_fd_coverage,
            "fd_coverage": fd_coverage,
        }
