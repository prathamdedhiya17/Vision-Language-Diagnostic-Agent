from ultralytics import YOLO
from PIL import Image
from config import YOLO_MODEL_PATH, YOLO_CONFIDENCE, DEFECT_LABELS

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"[YOLO] Loading {YOLO_MODEL_PATH}...")
        _model = YOLO(YOLO_MODEL_PATH)
    return _model

def run_inference(image: Image.Image) -> list[dict]:
    results = get_model().predict(image, conf=YOLO_CONFIDENCE, verbose=False)[0]
    return [
        {
            "defect_type": DEFECT_LABELS.get(int(box.cls[0]), f"Class_{int(box.cls[0])}"),
            "confidence":  round(float(box.conf[0]), 3),
            "bbox":        [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in results.boxes
    ]
