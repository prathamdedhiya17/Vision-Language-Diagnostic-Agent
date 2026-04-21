from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"[Embed] Loading {EMBED_MODEL}...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def encode(texts: list[str]) -> list:
    return get_model().encode(texts).tolist()
