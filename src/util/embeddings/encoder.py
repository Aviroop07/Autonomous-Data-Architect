import threading
from pathlib import Path
from typing import List, Optional, Any
import numpy as np

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_MODEL_DIR_BASE = Path(__file__).resolve().parent.parent.parent.parent / "models"

# Global singleton model cache
_model: Optional[Any] = None
_model_lock = threading.Lock()
_model_name: Optional[str] = None

def _get_model(model_name: Optional[str] = None) -> Any:
    global _model, _model_name
    resolved = model_name or DEFAULT_MODEL
    if _model is None or _model_name != resolved:
        with _model_lock:
            if _model is None or _model_name != resolved:
                from sentence_transformers import SentenceTransformer
                local_path = _MODEL_DIR_BASE / resolved
                if local_path.is_dir():
                    _model = SentenceTransformer(str(local_path))
                else:
                    _model = SentenceTransformer(resolved)
                _model_name = resolved
    return _model

def embed_texts(texts: List[str], batch_size: int = 32, model_name: Optional[str] = None) -> np.ndarray:
    if not texts:
        return np.array([])
    model = _get_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    return embeddings
