"""
test_api.py — Quick test for the VLDA Flask API
Usage: python test_api.py path/to/pcb_image.jpg
"""

import sys
import json
import requests

API_URL = "http://localhost:5000"

def test_health():
    r = requests.get(f"{API_URL}/health")
    print("Health:", json.dumps(r.json(), indent=2))

def test_analyze(image_path: str):
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_URL}/analyze",
            files={"image": (image_path, f, "image/jpeg")}
        )

    result = r.json()

    print(f"\n── Detections ({len(result['detections'])}) ──")
    for d in result["detections"]:
        print(f"  {d['defect_type']} | conf: {d['confidence']} | bbox: {d['bbox']}")

    print(f"\n── IPC Chunks Retrieved ──")
    for defect, chunks in result["rag_chunks"].items():
        print(f"  {defect}: {len(chunks)} chunks (top relevance: {chunks[0]['relevance']})")

    print(f"\n── Diagnostic Report ──\n")
    print(result["report"])

if __name__ == "__main__":
    test_health()
    if len(sys.argv) > 1:
        test_analyze(sys.argv[1])
    else:
        print("\nProvide an image path to test /analyze: python test_api.py image.jpg")