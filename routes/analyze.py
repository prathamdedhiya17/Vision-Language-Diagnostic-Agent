import io, base64
from flask import Blueprint, request, jsonify
from PIL import Image

from models          import yolo_model
from services        import rag_service, report_service
from services.evaluation import ragas_eval, llm_judge, citation_verify

analyze_bp = Blueprint("analyze", __name__)


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    # Parse image
    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")
    elif request.is_json and "image_base64" in request.json:
        image = Image.open(
            io.BytesIO(base64.b64decode(request.json["image_base64"]))
        ).convert("RGB")
    else:
        return jsonify({"error": "Provide 'image' as form-data or 'image_base64' as JSON."}), 400

    # YOLO
    detections = yolo_model.run_inference(image)
    if not detections:
        return jsonify({
            "detections": [], "defect_cards": [],
            "verdict": "ACCEPT",
            "verdict_reason": "No defects detected above confidence threshold.",
            "evaluation": {},
        })

    # RAG
    rag_contexts = rag_service.retrieve_all(detections)

    # Report
    raw_report = report_service.generate(detections, rag_contexts)
    parsed     = report_service.parse(raw_report, detections)

    # Evaluations
    evaluation = {
        "ragas":     ragas_eval.run(detections, rag_contexts, raw_report),
        "llm_judge": llm_judge.run(raw_report, detections, rag_contexts),
        "citations": citation_verify.run(parsed["defect_cards"]),
    }

    # Top-level pass: all three must pass
    evaluation["overall_pass"] = all([
        evaluation["ragas"].get("pass",     False),
        evaluation["llm_judge"].get("overall", {}).get("pass", False),
        evaluation["citations"].get("pass",  False),
    ])

    return jsonify({"detections": detections, "evaluation": evaluation, **parsed})
