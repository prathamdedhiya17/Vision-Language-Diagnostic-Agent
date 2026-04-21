import os
from dotenv import load_dotenv
load_dotenv()

# ── Paths (relative to Vision-Language-Diagnostic-Agent/) ─
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "YOLO/VLDA/yolov8n.pt")
CHROMA_DIR      = os.getenv("CHROMA_DIR",      "RAG/ipc_vectordb")
COLLECTION_NAME = "ipc_a_610g"

# ── Models ─────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ── Inference ──────────────────────────────────────────────
YOLO_CONFIDENCE = 0.25
TOP_K_CHUNKS    = 4

# ── Defect class map — must match YOLO/data.yaml order ────
DEFECT_LABELS = {
    0: "Open Circuit",
    1: "Short Circuit",
    2: "Mouse Bite",
    3: "Spur",
    4: "Spurious Copper",
    5: "Missing Hole",
}

# ── Evaluation thresholds ──────────────────────────────────
# RAGAs: below these = FAIL for that metric
RAGAS_RETRIEVAL_PASS   = 0.70   # context_precision, context_recall
RAGAS_GENERATION_PASS  = 0.70   # faithfulness, answer_relevancy

# Citation: chunk must score below this cosine distance to count as a hit
CITATION_DISTANCE_THRESHOLD = 0.40

# Citation: chunk must contain IPC acceptability language to be a valid support
IPC_ACCEPTABILITY_KEYWORDS = [
    "target condition", "acceptable condition", "defect condition",
    "class 1", "class 2", "class 3",
    "shall", "must not", "not permitted", "acceptable", "reject",
    "workmanship", "acceptability", "inspection",
]
