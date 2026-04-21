import chromadb
from models.embed_model import encode
from config import CHROMA_DIR, COLLECTION_NAME, TOP_K_CHUNKS

_collection = None

def get_collection():
    global _collection
    if _collection is None:
        print(f"[RAG] Connecting to ChromaDB at {CHROMA_DIR}...")
        _collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
    return _collection

def retrieve(defect_type: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    results = get_collection().query(
        query_embeddings = encode([defect_type]),
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
    return [
        {
            "text":      doc,
            "page":      meta.get("page", "?"),
            "section":   meta.get("section", "general"),
            "relevance": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def retrieve_all(detections: list[dict]) -> dict:
    contexts = {}
    for det in detections:
        dt = det["defect_type"]
        if dt not in contexts:
            contexts[dt] = retrieve(dt)
    return contexts
