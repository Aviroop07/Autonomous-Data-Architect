import sys
import os
import numpy as np
from typing import Any, List, Set, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util as st_util
import nltk
from nltk.corpus import wordnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.pipeline.stage2.models.schema import Schema, Table


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
    base = dt.upper().split("(")[0].strip()
    return COARSE_DT_MAP.get(base, "TEXT")


class SchemaEvaluator:
    def __init__(self, sim_threshold: float = 0.6, lcs_threshold: float = 0.75) -> None:
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sim_threshold = sim_threshold
        self.lcs_threshold = lcs_threshold
        try:
            wordnet.ensure_loaded()
        except Exception:
            nltk.download("wordnet")

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

    def is_synonym(self, pred: str, gt: str) -> bool:
        synonyms: Set[str] = set()
        for syn in wordnet.synsets(pred):
            if syn is None:
                continue
            for lemma in syn.lemmas():  # type: ignore[union-attr]
                synonyms.add(lemma.name().lower())
        return gt.lower() in synonyms

    def get_similarity(self, s1: str, s2: str) -> float:
        emb1 = self.sim_model.encode(s1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(s2, convert_to_tensor=True)
        return float(st_util.cos_sim(emb1, emb2)[0][0])

    def match_names(self, pred: str, gt: str) -> bool:
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
        if not gt_set and not pred_set:
            return 1.0, 1.0
        if not gt_set or not pred_set:
            return 0.0, 0.0
        matched_gt: Set[str] = set()
        matched_pred: Set[str] = set()
        for p in pred_set:
            for g in gt_set:
                if g not in matched_gt and match_func(p, g):
                    matched_gt.add(g)
                    matched_pred.add(p)
                    break
        intersection_size = len(matched_gt)
        precision = intersection_size / len(pred_set) if pred_set else 0.0
        recall = intersection_size / len(gt_set) if gt_set else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        acc = 1.0 if f1 == 1.0 else 0.0
        return f1, acc

    def evaluate_schema(
        self,
        pred_schema: Schema,
        gt_schema: Schema,
        gt_col_types: Optional[Dict[str, str]] = None,
        pred_col_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # 1. Table F1
        gt_table_names = {t.name for t in gt_schema.tables}
        pred_table_names = {t.name for t in pred_schema.tables}
        table_f1, table_acc = self.calculate_f1(
            pred_table_names, gt_table_names, self.match_names
        )

        # 2. Align tables
        matched_tables: List[Tuple[Table, Table]] = []
        remaining_gt = list(gt_schema.tables)
        for pt in pred_schema.tables:
            for gt in remaining_gt:
                if self.match_names(pt.name, gt.name):
                    matched_tables.append((gt, pt))
                    remaining_gt.remove(gt)
                    break

        # 3. Attribute F1 (averaged over GT tables)
        attr_f1_scores: List[float] = []
        for gt_t in gt_schema.tables:
            pred_t = next((p for g, p in matched_tables if g.name == gt_t.name), None)
            if pred_t:
                gt_attrs = {c.name for c in gt_t.columns}
                pred_attrs = {c.name for c in pred_t.columns}
                f1_attr, _ = self.calculate_f1(pred_attrs, gt_attrs, self.match_names)
                attr_f1_scores.append(f1_attr)
            else:
                attr_f1_scores.append(0.0)
        avg_attr_f1 = float(np.mean(attr_f1_scores)) if attr_f1_scores else 0.0
        attr_acc = 1.0 if avg_attr_f1 == 1.0 else 0.0

        # 4. PK accuracy -- exact match per Text2Schema protocol
        pk_correct = 0
        for gt_t in gt_schema.tables:
            pred_t = next((p for g, p in matched_tables if g.name == gt_t.name), None)
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

        fk_correct = 0
        for gt_t in gt_schema.tables:
            pred_t = next((p for g, p in matched_tables if g.name == gt_t.name), None)
            if pred_t:
                gt_fks = get_fk_set(gt_schema, gt_t)
                pred_fks_raw = get_fk_set(pred_schema, pred_t)
                pred_fks_mapped: Set[Tuple[str, str]] = set()
                for p_col, p_ref_table in pred_fks_raw:
                    gt_ref_table = next(
                        (g.name for g, p in matched_tables if p.name == p_ref_table),
                        p_ref_table,
                    )
                    gt_match = next(
                        (g for g, p in matched_tables if p.name == pred_t.name), None
                    )
                    gt_mapped_col = p_col
                    if gt_match:
                        for g_col_obj in gt_match.columns:
                            if self.match_names(p_col, g_col_obj.name):
                                gt_mapped_col = g_col_obj.name
                                break
                    pred_fks_mapped.add((gt_mapped_col, gt_ref_table))
                if gt_fks == pred_fks_mapped:
                    fk_correct += 1
        fk_acc = fk_correct / len(gt_schema.tables) if gt_schema.tables else 1.0

        # 6. DT accuracy -- coarse 5-category, requires explicit type maps
        dt_acc: Optional[float]
        if gt_col_types is not None and pred_col_types is not None:
            dt_correct = 0
            dt_total = 0
            for gt_t in gt_schema.tables:
                pred_t = next(
                    (p for g, p in matched_tables if g.name == gt_t.name), None
                )
                for gt_col in gt_t.columns:
                    dt_total += 1
                    if pred_t is None:
                        continue
                    matched_pred_col: Optional[str] = None
                    for pc in pred_t.columns:
                        if self.match_names(pc.name, gt_col.name):
                            matched_pred_col = pc.name
                            break
                    if matched_pred_col is None:
                        continue
                    gt_dt = coarsen_dt(gt_col_types.get(f"{gt_t.name}.{gt_col.name}"))
                    pred_dt = coarsen_dt(
                        pred_col_types.get(f"{pred_t.name}.{matched_pred_col}")
                    )
                    if gt_dt == pred_dt:
                        dt_correct += 1
            dt_acc = dt_correct / dt_total if dt_total > 0 else 1.0
        else:
            dt_acc = None

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
            pred_t = next((p for g, p in matched_tables if g.name == gt_t.name), None)
            if pred_t:
                gt_pk_set = set(gt_t.primary_key)
                pred_pk_set = set(pred_t.primary_key)
                pk_f1, _ = self.calculate_f1(pred_pk_set, gt_pk_set, self.match_names)
                if pk_f1 == 1.0:
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
            "attr_coverage_f1": attr_coverage_f1,
            "pk_fd_coverage": pk_fd_coverage,
            "fk_fd_coverage": fk_fd_coverage,
            "fd_coverage": fd_coverage,
        }
