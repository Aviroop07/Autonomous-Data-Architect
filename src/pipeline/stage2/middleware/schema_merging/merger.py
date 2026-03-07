import random
from typing import List, Dict, Tuple, Optional
from src.pipeline.stage2.models.schema import SchemaSegment, Table, Column, ForeignKey
from src.util.matching import gale_shapley_matching
from src.pipeline.stage2.middleware.schema_merging.similarity import get_similarity_score, get_similarity_matrix

class SchemaMerger:
    def __init__(self, alpha: float = 0.5, col_thresh: float = 0.8, table_thresh: float = 0.8):
        self.alpha = alpha
        self.col_thresh = col_thresh
        self.table_thresh = table_thresh

    def merge_segments(self, segments: List[SchemaSegment]) -> SchemaSegment:
        """
        Merges multiple schema segments sequentially: ((A + B) + C) + ...
        """
        if not segments:
            return SchemaSegment(chunk_title="Empty Schema", tables=[])
        
        if len(segments) == 1:
            segments[0].normalize()
            return segments[0]
            
        result = segments[0]
        result.normalize()
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            next_segment.normalize()
            result = self.merge_two_segments(result, next_segment)
            
        return result

    def merge_two_segments(self, A: SchemaSegment, B: SchemaSegment) -> SchemaSegment:
        """
        Merges two schema segments A and B. Updates A and returns it.
        """
        # A and B are assumed to be normalized already
        
        # 1. Calculate table score matrix
        table_list_a = A.tables
        table_list_b = B.tables
        
        if not table_list_a:
            return B.model_copy(deep=True)
        if not table_list_b:
            return A
            
        score_matrix = []
        for t1 in table_list_a:
            row = []
            for t2 in table_list_b:
                score = self._calculate_table_score(t1, t2)
                row.append(score)
            score_matrix.append(row)
            
        # 2. Match tables
        table_matches = gale_shapley_matching(score_matrix, self.table_thresh)
        
        matched_indices_a = {m[0] for m in table_matches}
        matched_indices_b = {m[1] for m in table_matches}
        
        # 3. Merge matched tables
        for idx_a, idx_b in table_matches:
            t1 = table_list_a[idx_a]
            t2 = table_list_b[idx_b]
            self._merge_tables(A, t1, t2)
            
        # 4. Add unmatched tables from B to A
        for i, t2 in enumerate(table_list_b):
            if i not in matched_indices_b:
                # Need to check if a table with the same name already exists in A (unlikely after normalization unless they weren't matched)
                if not any(t.name == t2.name for t in A.tables):
                    new_table = t2.model_copy(deep=True)
                    A.tables.append(new_table)
                    
        # 5. Merge relationships
        if B.relationships:
            if A.relationships is None:
                A.relationships = []
            
            existing_rels = set()
            for rel in A.relationships:
                existing_rels.add((rel.referencing_table, rel.referencing_column, rel.referred_table))
                
            for rel in B.relationships:
                # Note: Table names and column names might have changed during matching?
                # The user said: "for any pair of matched tables, the merged table will have the name of either one of the original tables."
                # My implementation keeps A's table name.
                # If a table in B was matched to a table in A, its name in relationships from B should be renamed to A's table name.
                
                # However, normalization already happened. If they matched, they might still have different names.
                # Let's handle renaming.
                pass # Wait, let's implement the renaming logic properly.

        # Re-handle relationships with mapping
        table_map_b_to_a = {}
        for idx_a, idx_b in table_matches:
            table_map_b_to_a[table_list_b[idx_b].name] = table_list_a[idx_a].name
            
        if B.relationships:
            if A.relationships is None:
                A.relationships = []
                
            for rel in B.relationships:
                new_rel = rel.model_copy(deep=True)
                # Remap tables
                new_rel.referencing_table = table_map_b_to_a.get(new_rel.referencing_table, new_rel.referencing_table)
                new_rel.referred_table = table_map_b_to_a.get(new_rel.referred_table, new_rel.referred_table)
                
                # Check for uniqueness before adding
                exists = False
                for r in A.relationships:
                    if (r.referencing_table == new_rel.referencing_table and 
                        r.referencing_column == new_rel.referencing_column and 
                        r.referred_table == new_rel.referred_table):
                        exists = True
                        break
                if not exists:
                    A.relationships.append(new_rel)
                    
        return A

    def _calculate_table_score(self, t1: Table, t2: Table) -> float:
        name_score = get_similarity_score(t1.name, t2.name)
        
        # Column matching
        cols1 = [c.name for c in t1.columns]
        cols2 = [c.name for c in t2.columns]
        
        col_score_matrix = get_similarity_matrix(cols1, cols2)
        col_matches = gale_shapley_matching(col_score_matrix, self.col_thresh)
        
        attr_score = sum(col_score_matrix[m[0]][m[1]] for m in col_matches)
        
        return self.alpha * name_score + (1.0 - self.alpha) * attr_score

    def _merge_tables(self, segment_a: SchemaSegment, t1: Table, t2: Table):
        """
        Merges t2 into t1 within segment_a.
        """
        cols1 = [c.name for c in t1.columns]
        cols2 = [c.name for c in t2.columns]
        
        col_score_matrix = get_similarity_matrix(cols1, cols2)
        col_matches = gale_shapley_matching(col_score_matrix, self.col_thresh)
        
        matched_indices_a = {m[0] for m in col_matches}
        matched_indices_b = {m[1] for m in col_matches}
        
        # Merge matched columns
        for idx_a, idx_b in col_matches:
            c1 = t1.columns[idx_a]
            c2 = t2.columns[idx_b]
            # Pick a name randomly from either one
            new_name = random.choice([c1.name, c2.name])
            if new_name != c1.name:
                segment_a.rename_column(t1.name, c1.name, new_name)
        
        # Add unmatched columns from t2 to t1
        existing_col_names = {c.name for c in t1.columns}
        for i, c2 in enumerate(t2.columns):
            if i not in matched_indices_b:
                if c2.name not in existing_col_names:
                    t1.columns.append(c2.model_copy(deep=True))
                    existing_col_names.add(c2.name)
        
        # Merge unique constraints
        if t2.unique:
            if t1.unique is None:
                t1.unique = []
            for uq2 in t2.unique:
                # Check for existing
                exists = False
                for uq1 in t1.unique:
                    if set(uq1.columns) == set(uq2.columns):
                        exists = True
                        break
                if not exists:
                    t1.unique.append(uq2.model_copy(deep=True))
