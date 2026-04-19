"""
IPC-A-610G → ChromaDB RAG Pipeline (Text Only)

pip install pymupdf sentence-transformers chromadb
"""

import fitz
import json
import chromadb
from sentence_transformers import SentenceTransformer

# ── CONFIG ──────────────────────────────────────────────
PDF_PATH     = "IPC-A-610G_2017.pdf"
CHROMA_DIR   = "./ipc_vectordb"
COLLECTION   = "ipc_a_610g"
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE   = 400   # words
OVERLAP      = 50    # words

DEFECT_KEYWORDS = [
    "open circuit", "short circuit", "mouse bite", "spur",
    "spurious copper", "missing hole", "solder bridge",
    "cold solder", "lifted pad", "excess solder", "void"
]


# ── STEP 1: EXTRACT TEXT ─────────────────────────────────
def extract_text(pdf_path: str) -> list[dict]:
    print("[1/3] Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append({"page": page.number + 1, "text": text})
    doc.close()
    print(f"      Extracted text from {len(pages)} pages.")
    return pages


# ── STEP 2: CHUNK ────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    print("[2/3] Chunking...")
    chunks = []
    for entry in pages:
        words = entry["text"].split()
        page  = entry["page"]
        for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
            chunk_words = words[i : i + CHUNK_SIZE]
            if len(chunk_words) < 20:
                continue
            text = " ".join(chunk_words)
            chunks.append({
                "id":      f"p{page}_c{i}",
                "text":    text,
                "page":    page,
                "defects": json.dumps([k for k in DEFECT_KEYWORDS if k in text.lower()]),
                "section": infer_section(text)
            })
    print(f"      Created {len(chunks)} chunks.")
    return chunks


def infer_section(text: str) -> str:
    t = text.lower()
    if any(c in t for c in ["class 1", "class 2", "class 3"]):
        return "acceptance_criteria"
    if any(w in t for w in ["repair", "rework", "remediat"]):
        return "remediation"
    if "root cause" in t or "cause" in t:
        return "root_cause"
    if "solder" in t:
        return "soldering"
    return "general"


# ── STEP 3: EMBED + STORE ────────────────────────────────
def embed_and_store(chunks: list[dict]) -> None:
    print(f"[3/3] Embedding and storing {len(chunks)} chunks...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        batch_size=32
    )

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids        = [c["id"] for c in chunks],
        documents  = [c["text"] for c in chunks],
        embeddings = [e.tolist() for e in embeddings],
        metadatas  = [{"page": c["page"], "defects": c["defects"], "section": c["section"]} for c in chunks]
    )
    print(f"      Done. {collection.count()} chunks stored in '{CHROMA_DIR}'.")


# ── QUERY (called at runtime by YOLO pipeline) ───────────
def query_ipc(defect_type: str, top_k: int = 3) -> list[dict]:
    model      = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION)

    results = collection.query(
        query_embeddings = model.encode([defect_type]).tolist(),
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":      doc,
            "page":      meta["page"],
            "section":   meta["section"],
            "relevance": round(1 - dist, 3)
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


# ── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Build once
    pages  = extract_text(PDF_PATH)
    chunks = chunk_pages(pages)
    embed_and_store(chunks)
    print("\n✅ Knowledge base ready.\n")

    # Test query — simulates YOLO detecting a defect
    print("── Test query: 'open circuit' ──")
    for i, r in enumerate(query_ipc("open circuit"), 1):
        print(f"\n[{i}] Page {r['page']} | {r['section']} | relevance: {r['relevance']}")
        print(r["text"][:300] + "...")
