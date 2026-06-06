from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.stage2.models.schema import Schema
    from src.pipeline.stage3.models.nodes import JoinNode, AggNode
    # Full union type for type-checkers; forward refs are strings because
    # BaseTableNode is defined below and JoinNode/AggNode are in nodes.py.
    TableNode = Union["BaseTableNode", "JoinNode", "AggNode"]
else:
    # Runtime alias — Pydantic resolves field annotations lazily, so Any
    # is safe here. nodes.py re-exports a fully-resolved TableNode after
    # all three concrete classes are defined.
    TableNode = Any

class RelationalCycleError(Exception):
    """Custom error for circular dependencies in the relational graph."""
    # node_stack holds mixed node types (BaseTableNode/JoinNode/AggNode/etc.)
    # defined across base.py and nodes.py; Any is intentional here to avoid
    # a circular import between base.py and nodes.py.
    def __init__(self, node_stack: List[Any]):
        steps = []
        for i, n in enumerate(node_stack):
            is_reentry = (i == len(node_stack) - 1)
            prefix = f"  [{i+1}] " if not is_reentry else "  >> RE-ENTRY: "
            if hasattr(n, 'describe'): desc = n.describe()
            else:
                desc = f"{type(n).__name__}"
                if hasattr(n, 'name'): desc += f"({n.name})"
            steps.append(f"{prefix}{desc}")
        path_str = "\n".join(steps)
        message = f"CRITICAL: RELATIONAL RECURSION DETECTED\n{path_str}"
        super().__init__(message)

class Column(BaseModel):
    name: str

class BaseTableNode(BaseModel):
    """A physical table in the database schema."""
    name: str = Field(description="The formal name of the table.")
    alias: Optional[str] = None

    def describe(self) -> str: return f"Table '{self.name}'"
    def get_columns(self, schema: Any) -> Set[str]:
        t_map = {t.name: t for t in schema.tables}
        if self.name not in t_map: return set()
        return {c.name for c in t_map[self.name].columns}
    def get_signature(self, schema: Any = None) -> str: return f"Base:{self.name}:{self.alias or ''}"
    def get_relational_components(self, schema: Any) -> 'Tuple[Set[str], Set[str]]': return {self.get_signature()}, set()
    def unify(self, registry: Dict[str, Any], schema: Any, path: Optional[List[Any]] = None) -> 'BaseTableNode':
        if path is None: path = []
        if any(id(self) == id(n) for n in path): raise RelationalCycleError(path + [self])
        sig = self.get_signature()
        if sig in registry: return registry[sig]
        registry[sig] = self
        return self
    def get_identity_col(self, schema: Any) -> str:
        t_map = {t.name: t for t in schema.tables}
        if self.name not in t_map: return "id"
        return getattr(t_map[self.name], 'pk', 'id')
    def find_origin_table(self, column_name: str, schema: Any) -> str: return self.get_signature()
    def get_column_type(self, column_name: str, schema: Any) -> Optional[str]:
        t_map = {t.name.upper(): t for t in schema.tables}
        if self.name.upper() not in t_map: return None
        for c in t_map[self.name.upper()].columns:
            if c.name.lower() == column_name.lower():
                return c.data_type
        return None

    def _validate(self, schema: Any) -> List[str]:
        t_map = {t.name.upper(): t for t in schema.tables}
        return [] if self.name.upper() in t_map else [f"Table '{self.name}' not found."]
