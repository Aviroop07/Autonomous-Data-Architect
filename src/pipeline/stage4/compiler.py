import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Union
from src.pipeline.stage3.models import (
    AlgebraicManifest, IfNode, LogicalNode,
    CombinationNode, UnivariateDist, NumericRange, DateRange,
    NormalDist, LogNormalDist, PoissonDist, ZipfDist, CategoricalDist
)
from src.pipeline.stage3.models.nodes import ColumnNode, ConstNode
from src.pipeline.stage4.models import ParameterManifest, TableParameters

class MinimalCompiler:
    """
    Deterministic Synthesis Engine.
    Emits vectorized Python/Pandas logic based on algebraic metadata and parameter scales.
    """
    def __init__(
        self,
        manifest: AlgebraicManifest,
        parameters: ParameterManifest,
        enable_logical_constraints: bool = True,
    ):
        self.manifest = manifest
        self.enable_logical_constraints = enable_logical_constraints
        # Convert List to Dict for O(1) lookup
        self.param_map = {p.table_name: p for p in parameters.parameters}
        self.code_lines = []
        self._indent = 0
        self._current_table = None
        self.schema = None
    def _get_pk(self, table_name: str, schema: Dict[str, Any]) -> str:
        for t in schema['tables']:
            if t['name'] == table_name:
                return t.get('pk', 'id')
        return 'id'

    def emit(self, line: str) -> None:
        self.code_lines.append("    " * self._indent + line)

    def compile(self, schema_json: str) -> str:
        self.schema = json.loads(schema_json)
        schema = self.schema
        self.emit("import pandas as pd")
        self.emit("import numpy as np")
        self.emit("import random")
        self.emit("from datetime import datetime")
        self.emit("")
        self.emit("SCALE_FACTOR = 1.0")
        self.emit("")
        self.emit("BUFFERS = {}")
        self.emit("")

        # 1. Topological Sort for Table Generation
        gen_order = self._get_topological_order(schema)

        for table_name in gen_order:
            self._current_table = table_name
            self._compile_table(table_name, schema)

        # 2. Apply Global Constraints (if any) — skipped when logical constraints are disabled
        global_rules = getattr(self.manifest, "global_rules", [])
        if self.enable_logical_constraints and global_rules:
            self.emit("")
            self.emit("# --- GLOBAL CONSTRAINTS ---")
            for rule in global_rules:
                if not isinstance(rule, IfNode):
                    continue
                t_target: Optional[str] = None
                if hasattr(rule.anchor_table, "name"):
                    t_target = rule.anchor_table.name.upper()
                elif hasattr(rule.anchor_table, "l_table") and hasattr(  # type: ignore[union-attr]
                    rule.anchor_table.l_table, "name"  # type: ignore[union-attr]
                ):
                    t_target = rule.anchor_table.l_table.name.upper()  # type: ignore[union-attr]

                if t_target and t_target in [t["name"] for t in schema["tables"]]:
                    self._compile_if_node(rule, t_target)
                else:
                    if hasattr(rule.anchor_table, "l_table"):  # type: ignore[union-attr]
                        fallback = rule.anchor_table.l_table.name.upper()  # type: ignore[union-attr]
                        if fallback in [t["name"] for t in schema["tables"]]:
                            self._compile_if_node(rule, fallback)
                            continue
                    self.emit(f"# SKIPPED GLOBAL RULE: UNABLE TO RESOLVE ANCHOR {t_target}")

        self.emit("")
        self.emit("# --- EXPORT ---")
        self.emit("for name, df in BUFFERS.items():")
        self.emit("    df.to_csv(f'{name}.csv', index=False)")

        return "\n".join(self.code_lines)

    def _get_topological_order(self, schema: Dict[str, Any]) -> List[str]:
        tables = [t['name'] for t in schema['tables']]
        order = []
        visited = set()

        def visit(u):
            if u not in visited:
                visited.add(u)
                # Dependencies: find tables referred to by u
                for rel in (schema.get('relationships') or []):
                    if rel['referencing_table'] == u:
                        visit(rel['referred_table'])
                if u not in order:
                    order.append(u)

        for t in tables:
            visit(t)
        return order

    def _compile_table(self, table_name: str, schema: Dict[str, Any]) -> None:
        self.emit(f"# --- TABLE: {table_name} ---")

        # A. Determine Row Count
        param = self.param_map.get(table_name)
        pk = self._get_pk(table_name, schema)

        if param and param.n_seeds is not None:
            self.emit(f"BUFFERS['{table_name}'] = pd.DataFrame()")
            self.emit(f"BUFFERS['{table_name}']['{pk}'] = range(1, max(2, int({param.n_seeds} * SCALE_FACTOR)) + 1)")
        else:
            # Dependent Table (Fanout logic)
            parents = []
            for rel in (schema.get('relationships') or []):
                if rel['referencing_table'] == table_name:
                    parents.append(rel['referred_table'])

            if not parents:
                # Isolated table defaults
                self.emit(f"BUFFERS['{table_name}'] = pd.DataFrame()")
                self.emit(f"BUFFERS['{table_name}']['{pk}'] = range(1, 101)")
            else:
                fanout = param.avg_fanout if param and param.avg_fanout else 3.0
                parent = parents[0]
                self.emit(f"n_rows = int(len(BUFFERS['{parent}']) * {fanout})")
                self.emit(f"BUFFERS['{table_name}'] = pd.DataFrame()")
                self.emit(f"BUFFERS['{table_name}']['{pk}'] = range(1, n_rows + 1)")

        # LINK ALL FKS (Mandatory for Logic Readiness)
        for rel in (schema.get('relationships') or []):
            if rel['referencing_table'] == table_name:
                p_table = rel['referred_table']
                p_col = rel.get('referred_column') or self._get_pk(p_table, schema)
                r_col = rel['referencing_column']
                # Emission: Population from Parent
                if p_table in [t['name'] for t in schema['tables']]:
                     self.emit(f"BUFFERS['{table_name}']['{r_col}'] = np.random.choice(BUFFERS['{p_table}']['{p_col}'], size=len(BUFFERS['{table_name}']))")

        # B. Base Initialization for all other Columns (Ensures logic-execution readiness)
        # We initialize every column not already set (PK/FK) to random defaults
        if table_name in [t['name'] for t in schema['tables']]:
            curr_tab = next(t for t in schema['tables'] if t['name'] == table_name)
            for col in curr_tab['columns']:
                c_name = col['name']
                # Skip if already set (PK handled above, FKs handled above)
                if c_name == pk: continue
                is_fk = False
                for rel in (schema.get('relationships') or []):
                    if rel['referencing_table'] == table_name and rel['referencing_column'] == c_name:
                        is_fk = True; break
                if is_fk: continue

                # Check for explicit bounds in manifest FIRST to avoid logic-breaking defaults
                manifest = self.manifest.get_table_manifest(table_name)
                numeric_bounds = getattr(manifest, "numeric_bounds", {}) if manifest else {}
                bound = numeric_bounds.get(c_name) if manifest else None

                # Heuristic: If we have a numeric bound, prioritize numeric initialization even if data_type says otherwise
                is_numeric_bound = bound and (bound.min is not None or bound.max is not None)

                # Default base: random numbers or discrete ints
                if col['data_type'] in ['INT', 'BIGINT', 'SMALLINT'] or is_numeric_bound:
                    low = int(bound.min) if bound and bound.min is not None else 0
                    high = int(bound.max) if bound and bound.max is not None else 100
                    if bound is None or (bound.min is None and bound.max is None):
                        c_lower = c_name.lower()
                        if "score" in c_lower: low, high = 300, 850
                    if low >= high: high = low + 100 # Safety
                    self.emit(f"BUFFERS['{table_name}']['{c_name}'] = np.random.randint({low}, {high}, size=len(BUFFERS['{table_name}']))")
                elif col['data_type'] in ['FLOAT', 'DOUBLE', 'DECIMAL', 'REAL']:
                    low = float(bound.min) if bound and bound.min is not None else 0.0
                    high = float(bound.max) if bound and bound.max is not None else 1.0
                    # Domain-aware scale fallback for common financial columns if bound is missing
                    if bound is None or (bound.min is None and bound.max is None):
                        c_lower = c_name.lower()
                        if "amount" in c_lower or "principal" in c_lower: low, high = 1000.0, 50000.0
                        elif "income" in c_lower or "salary" in c_lower: low, high = 20000.0, 100000.0
                        elif "rate" in c_lower or "interest" in c_lower: low, high = 1.0, 15.0

                    if low >= high: high = low + 1.0 # Safety
                    self.emit(f"BUFFERS['{table_name}']['{c_name}'] = np.random.uniform({low}, {high}, size=len(BUFFERS['{table_name}']))")
                else:
                    self.emit(f"BUFFERS['{table_name}']['{c_name}'] = [''] * len(BUFFERS['{table_name}'])")

        # C. Apply Distributions (Overrides base)
        manifest = self.manifest.get_table_manifest(table_name)
        if manifest:
            distributions = getattr(manifest, "distributions", {})
            numeric_bounds = getattr(manifest, "numeric_bounds", {})
            for col, dist in distributions.items():
                sampler_code = self._emit_dist_sampler(dist, table_name)
                self.emit(f"BUFFERS['{table_name}']['{col}'] = {sampler_code}")
                # Clipping Bounds (New NumericRange model)
                if col in numeric_bounds:
                    b = numeric_bounds[col]
                    low = b.min if b.min is not None else -np.inf
                    high = b.max if b.max is not None else np.inf
                    if low != -np.inf or high != np.inf:
                        l_v = "np.NINF" if np.isinf(low) and low < 0 else str(low)
                        h_v = "np.inf" if np.isinf(high) and high > 0 else str(high)
                        self.emit(f"BUFFERS['{table_name}']['{col}'] = np.clip(BUFFERS['{table_name}']['{col}'], {l_v}, {h_v})")

        # D. SEMANTIC INFILL PLACEHOLDER (Crucial for rule predictive readiness)
        self.emit(f"# --- SEMANTIC_INFILL_PLACEHOLDER_{table_name} ---")

        # E. Apply Logical Predicates (Overrides Distributions and Infill)
        if manifest and self.enable_logical_constraints:
            for rule in getattr(manifest, "logical_rules", []):
                self._compile_if_node(rule, table_name)

        # D. Apply Sparsity (Dropping Mechanism)
        if param and hasattr(param, 'sparsity'):
            for col, prob in param.sparsity.items():
                if prob > 0:
                    self.emit(f"# Nullability Sparsity for {col}")
                    self.emit(f"BUFFERS['{table_name}']['{col}'] = BUFFERS['{table_name}']['{col}'].mask(np.random.random(len(BUFFERS['{table_name}'])) < {prob}, np.nan)")

    def _emit_dist_sampler(self, dist: UnivariateDist, table_name: str) -> str:
        d = dist.distribution
        if isinstance(d, NormalDist):
            return f"np.random.normal({d.mean}, np.sqrt({d.variance}), size=len(BUFFERS['{table_name}']))"
        elif isinstance(d, PoissonDist):
            return f"np.random.poisson({d.lam}, size=len(BUFFERS['{table_name}']))"
        elif isinstance(d, ZipfDist):
            return f"(np.random.zipf({d.a}, size=len(BUFFERS['{table_name}'])).astype(int))"
        elif isinstance(d, CategoricalDist):
            labels = list(d.weights.keys())
            probs = list(d.weights.values())
            total = sum(probs)
            probs = [p/total for p in probs]
            return f"np.random.choice({labels}, p={probs}, size=len(BUFFERS['{table_name}']))"
        elif isinstance(d, LogNormalDist):
            return f"np.random.lognormal({d.mean}, np.sqrt({d.variance}), size=len(BUFFERS['{table_name}']))"
        return "np.nan"

    def _compile_if_node(self, node: IfNode, table_name: str) -> None:
        for pair in node.pairs:
            cond_mask = self._emit_logic_mask(pair.condition, table_name)
            self.emit(f"mask = {cond_mask}")
            self._apply_result(pair.result, table_name, "mask")

    def _apply_result(self, node: Any, table_name: str, mask_var: str) -> None:
        """Recursively applies assignments from an algebraic result node."""
        if hasattr(node, "operator") and node.operator == "IS_NULL":
            col = node.column_1.name if hasattr(node.column_1, 'name') else 'unknown'
            self.emit(f"BUFFERS['{table_name}'].loc[{mask_var}, '{col}'] = np.nan")
        elif hasattr(node, "column_1") and hasattr(node, "column_2"):
            # Leaf Assignment (LogicalNode)
            col = node.column_1.name if hasattr(node.column_1, 'name') else 'unknown'

            # Use resolve_col to handle parent references in assignments
            val = self._resolve_rhs(node.column_2, table_name)

            self.emit(f"BUFFERS['{table_name}'].loc[{mask_var}, '{col}'] = {val}")
        elif hasattr(node, "operands"):
            # Recursive CombinationNode
            for op in node.operands:
                self._apply_result(op, table_name, mask_var)

    def _resolve_rhs(self, node: Union[ColumnNode, ConstNode], table_name: str) -> str:
        """Helper to resolve the right-hand side of an assignment."""
        if not hasattr(node, 'name'):
            val = getattr(node, 'value', node)
            # String constants must be repr()'d so they emit as 'value', not bare identifier
            if isinstance(val, str):
                return repr(val)
            return str(val)

        # It's a column, resolve it (potentially from parent)
        return self._emit_logic_mask_expr(node, table_name)

    def _emit_logic_mask_expr(self, col_node: Any, current_table: str) -> str:
        """Helper to resolve a single column into a pandas expression (reused by mask and apply)."""
        assert self.schema is not None, "_emit_logic_mask_expr called before compile()"
        c_name = col_node.name
        # 1. Check current table
        for t in (self.schema.get('tables') or []):
            if t['name'] == current_table:
                if any(c['name'] == c_name for c in t['columns']):
                    return f"BUFFERS['{current_table}']['{c_name}']"

        # 2. Check parents
        for rel in (self.schema.get('relationships') or []):
            if rel['referencing_table'] == current_table:
                p_table = rel['referred_table']
                r_col = rel['referencing_column']
                for t in (self.schema.get('tables') or []):
                    if t['name'] == p_table:
                        if any(c['name'] == c_name for c in t['columns']):
                            p_col = rel.get('referred_column') or self._get_pk(p_table, self.schema)
                            return f"BUFFERS['{current_table}']['{r_col}'].map(BUFFERS['{p_table}'].set_index('{p_col}')['{c_name}'])"

        # Column not found in current table or parents — fall back to PK
        # so any self-comparison (pk == pk) becomes always-True rather than crashing
        pk = self._get_pk(current_table, self.schema)
        return f"BUFFERS['{current_table}']['{pk}']"

    def _emit_logic_mask(self, node: Any, table_name: str) -> str:
        if hasattr(node, "operator") and hasattr(node, "column_1"):
            c1 = self._emit_logic_mask_expr(node.column_1, table_name)

            if node.operator == "IS_NULL":
                return f"{c1}.isna()"
            if node.operator == "IS_NOT_NULL":
                return f"{c1}.notna()"

            c2 = self._resolve_rhs(node.column_2, table_name)
            op_map = {"EQUALS": "==", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<=", "NOT_EQUALS": "!="}
            return f"({c1} {op_map.get(node.operator, '==')} {c2})"
        elif hasattr(node, "operator") and hasattr(node, "operands"):
            ops = [self._emit_logic_mask(o, table_name) for o in node.operands]
            conj = " & " if node.operator == "AND" else " | "
            return f"({conj.join(ops)})"
        return "True"
