import random
from typing import List, Dict, Tuple, Optional
from src.pipeline.stage2.models.schema import Schema, Table, Column, ForeignKey
from src.util.matching import gale_shapley_matching
from src.pipeline.stage2.middleware.schema_merging.similarity import get_similarity_score, get_similarity_matrix

class SchemaMerger:
    def __init__(self, alpha: float = 0.5, col_thresh: float = 0.8, table_thresh: float = 0.8):
        self.alpha = alpha
        self.col_thresh = col_thresh
        self.table_thresh = table_thresh

    def merge_segments(self, segments: List[Schema], authoritative_dimensions: List[str] = None) -> Schema:
        """
        Merges multiple schema segments sequentially.
        If authoritative_dimensions are provided, it maintains their definitions as masters.
        """
        if not segments:
            return Schema(tables=[])
        
        self.authoritative_dimensions = {d.lower() for d in (authoritative_dimensions or [])}
        
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

    def merge_two_segments(self, A: Schema, B: Schema) -> Schema:
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
                # Need to check if a table with the same name already exists in A
                if not any(t.name == t2.name for t in A.tables):
                    new_table = t2.model_copy(deep=True)
                    A.tables.append(new_table)
                    
        # 5. Merge relationships
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

    def _merge_tables(self, schema_a: Schema, t1: Table, t2: Table):
        """
        Merges t2 into t1 within schema_a.
        Uses Union-based merging and Authoritative Priority.
        """
        cols1 = [c.name for c in t1.columns]
        cols2 = [c.name for c in t2.columns]
        
        col_score_matrix = get_similarity_matrix(cols1, cols2)
        col_matches = gale_shapley_matching(col_score_matrix, self.col_thresh)
        
        matched_indices_a = {m[0] for m in col_matches}
        matched_indices_b = {m[1] for m in col_matches}
        
        is_t1_authoritative = t1.name.lower() in self.authoritative_dimensions
        is_t2_authoritative = t2.name.lower() in self.authoritative_dimensions

        # Merge matched columns
        for idx_a, idx_b in col_matches:
            c1 = t1.columns[idx_a]
            c2 = t2.columns[idx_b]
            
            # Name priority: T1 > T2 unless only T2 is authoritative
            if is_t2_authoritative and not is_t1_authoritative:
                new_name = c2.name
            else:
                new_name = c1.name
                
            if new_name != c1.name:
                schema_a.rename_column(t1.name, c1.name, new_name)
        
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
                # Check for existing
                exists = False
                for uq1 in t1.unique:
                    if set(uq1.columns) == set(uq2.columns):
                        exists = True
                        break
                if not exists:
                    t1.unique.append(uq2.model_copy(deep=True))
