from flask import Blueprint, jsonify
from services.rag_service import get_collection
from config import YOLO_MODEL_PATH, CLAUDE_MODEL, EMBED_MODEL

health_bp = Blueprint("health", __name__)

@health_bp.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "yolo":       YOLO_MODEL_PATH,
        "embed":      EMBED_MODEL,
        "claude":     CLAUDE_MODEL,
        "ipc_chunks": get_collection().count(),
    })
