from typing import List, Set, Dict
from pydantic import BaseModel, Field

class TableFactRegistry(BaseModel):
    """
    Tracks which fact IDs contributed to which tables.
    This is kept OUTSIDE of the Schema LLM models to avoid distracting the agents.
    """
    table_to_facts: Dict[str, Set[int]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def register_table_facts(self, table_name: str, fact_ids: List[int]) -> None:
        table_name = table_name.upper()
        if table_name not in self.table_to_facts:
            self.table_to_facts[table_name] = set()
        self.table_to_facts[table_name].update(fact_ids)

    def merge_tables(self, source_table: str, target_table: str) -> None:
        """Merges facts from source_table into target_table and removes source_table."""
        source_table = source_table.upper()
        target_table = target_table.upper()
        if source_table in self.table_to_facts:
            source_facts = self.table_to_facts.pop(source_table)
            if target_table not in self.table_to_facts:
                self.table_to_facts[target_table] = set()
            self.table_to_facts[target_table].update(source_facts)

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Changes the table name in the registry while preserving associated facts."""
        old_name = old_name.upper()
        new_name = new_name.upper()
        if old_name in self.table_to_facts:
            facts = self.table_to_facts.pop(old_name)
            self.table_to_facts[new_name] = facts

    def delete_table(self, table_name: str) -> None:
        """Removes a table from the registry."""
        table_name = table_name.upper()
        if table_name in self.table_to_facts:
            self.table_to_facts.pop(table_name)

    def get_facts_for_tables(self, table_names: List[str]) -> Set[int]:
        union_set = set()
        for name in table_names:
            union_set.update(self.table_to_facts.get(name.upper(), set()))
        return union_set
