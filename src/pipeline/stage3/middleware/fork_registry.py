from pydantic import BaseModel
from typing import Literal, List, Dict, Optional
import sqlglot
from sqlglot import exp

class ForkKey(BaseModel):
    table_name: str
    column_name: str
    
    def __hash__(self):
        return hash((self.table_name, self.column_name))
        
    def to_string(self) -> str:
        return f"{self.table_name}.{self.column_name}"

class BranchCondition(BaseModel):
    fork_key: ForkKey
    operator: Literal["EQ", "NEQ", "IN"]
    values: List[str]

class ForkKeyRegistry:
    def __init__(self):
        self.forks: Dict[ForkKey, List[str]] = {} # fork_key -> list of categories
    
    def register_fork(self, fork_key: ForkKey, categories: List[str]):
        """Deduplicates and reconciles fork keys across shards."""
        if fork_key not in self.forks:
            self.forks[fork_key] = list(categories)
        else:
            existing = set(self.forks[fork_key])
            for cat in categories:
                if cat not in existing:
                    self.forks[fork_key].append(cat)
                    existing.add(cat)
            
    def get_branches_for_condition(self, condition: BranchCondition) -> List[str]:
        if condition.fork_key not in self.forks:
            return condition.values # Fallback
            
        all_cats = self.forks[condition.fork_key]
        if condition.operator == "EQ":
            return [v for v in all_cats if v in condition.values]
        elif condition.operator == "NEQ":
            return [v for v in all_cats if v not in condition.values]
        elif condition.operator == "IN":
            return [v for v in all_cats if v in condition.values]
        return []

def parse_if_condition(condition_str: str) -> Optional[BranchCondition]:
    try:
        parsed = sqlglot.parse_one(condition_str)
    except Exception as e:
        return None
        
    if isinstance(parsed, (exp.EQ, exp.NEQ)):
        left = parsed.left
        right = parsed.right
        
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            table = left.args.get('table')
            col = left.args.get('this')
            if not table or not col:
                return None
            
            table_name = table.name.upper()
            col_name = col.name.lower()
            val = right.name
            
            op = "EQ" if isinstance(parsed, exp.EQ) else "NEQ"
            return BranchCondition(
                fork_key=ForkKey(table_name=table_name, column_name=col_name),
                operator=op,
                values=[val]
            )
    elif isinstance(parsed, exp.In):
        left = parsed.this
        expressions = parsed.expressions
        if isinstance(left, exp.Column) and all(isinstance(e, exp.Literal) for e in expressions):
            table = left.args.get('table')
            col = left.args.get('this')
            if not table or not col:
                return None
                
            table_name = table.name.upper()
            col_name = col.name.lower()
            values = [e.name for e in expressions]
            
            return BranchCondition(
                fork_key=ForkKey(table_name=table_name, column_name=col_name),
                operator="IN",
                values=values
            )
    return None
