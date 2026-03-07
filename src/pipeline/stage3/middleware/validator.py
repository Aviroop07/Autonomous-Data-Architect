from typing import List, Dict, Set
from src.pipeline.stage2.models.schema import SchemaSegment

def detect_cycles(schema: SchemaSegment) -> List[List[str]]:
    """
    Detects cycles in the Foreign Key relationship graph.
    Returns a list of cycles, where each cycle is a list of table names.
    """
    if not schema.relationships:
        return []
    
    # Build adjacency list: table -> list of referred tables
    adj: Dict[str, Set[str]] = {}
    for rel in schema.relationships:
        if rel.referencing_table not in adj:
            adj[rel.referencing_table] = set()
        adj[rel.referencing_table].add(rel.referred_table)
        
    visited: Set[str] = set()
    stack: Set[str] = set()
    path: List[str] = []
    cycles: List[List[str]] = []
    
    def dfs(node: str):
        visited.add(node)
        stack.add(node)
        path.append(node)
        
        if node in adj:
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in stack:
                    # Cycle detected
                    # Extract the cycle from path
                    try:
                        idx = path.index(neighbor)
                        cycles.append(path[idx:] + [neighbor])
                    except ValueError:
                        pass
        
        stack.remove(node)
        path.pop()
        
    all_tables = {t.name for t in schema.tables}
    for table in all_tables:
        if table not in visited:
            dfs(table)
            
    return cycles
