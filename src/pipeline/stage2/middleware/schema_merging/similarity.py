from typing import List
from sentence_transformers import SentenceTransformer, util

class SemanticSimilarity:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_score(self, s1: str, s2: str) -> float:
        """
        Calculates the semantic similarity score between two strings.
        If they are exactly the same (case-insensitive), returns 1.0.
        Otherwise, returns the cosine similarity of their embeddings.
        """
        if s1.strip().lower() == s2.strip().lower():
            return 1.0
        
        embeddings = self.model.encode([s1, s2], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1])
        return float(cosine_score.item())

    def get_matrix_scores(self, list_a: List[str], list_b: List[str]) -> List[List[float]]:
        """
        Calculates a matrix of similarity scores between two lists of strings.
        """
        if not list_a or not list_b:
            return []
            
        # Optimization: encode all strings at once
        embeddings_a = self.model.encode(list_a, convert_to_tensor=True)
        embeddings_b = self.model.encode(list_b, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings_a, embeddings_b)
        
        matrix = cosine_scores.tolist()
        
        # Check for exact matches to override with 1.0
        for i, val_a in enumerate(list_a):
            for j, val_b in enumerate(list_b):
                if val_a.strip().lower() == val_b.strip().lower():
                    matrix[i][j] = 1.0
                    
        return matrix

# Singleton instance for convenience
_default_similarity = None

def get_similarity_score(s1: str, s2: str) -> float:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = SemanticSimilarity()
    return _default_similarity.get_score(s1, s2)

def get_similarity_matrix(list_a: List[str], list_b: List[str]) -> List[List[float]]:
    global _default_similarity
    if _default_similarity is None:
        _default_similarity = SemanticSimilarity()
    return _default_similarity.get_matrix_scores(list_a, list_b)
