import threading
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Global singleton model for efficiency
_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

def _get_model() -> SentenceTransformer:
    """Lazy loads the SentenceTransformer model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # all-MiniLM-L6-v2 is an 80MB model, fast and standard for local embeddings
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Computes dense vector embeddings for a batch of strings.
    
    Args:
        texts: A list of strings to embed.
        batch_size: The batch size for the encoder.
        
    Returns:
        A NumPy array of shape (len(texts), embedding_dim) containing the embeddings.
    """
    if not texts:
        return np.array([])
        
    model = _get_model()
    # SentenceTransformer.encode automatically handles batching and utilizes PyTorch vectorization
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return embeddings
