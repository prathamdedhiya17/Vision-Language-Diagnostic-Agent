# Vision-Language Diagnostic Agent (VLDA)

Automated PCB inspection that chains **YOLOv8 object detection**, **retrieval-augmented generation** over the IPC-A-610G standard, and **Claude** for report writing — with a three-way evaluation pipeline (RAGAs metrics, LLM-Judge G-Eval, citation verification) on every report.

---

## Overview

VLDA accepts a PCB image and produces a structured defect-by-defect inspection report grounded in IPC-A-610G acceptance criteria. The pipeline is:

1. **Detect** — A fine-tuned YOLOv8 model localises six classes of physical defects.
2. **Retrieve** — For each detected class, the top-K most relevant chunks of IPC-A-610G are pulled from a persistent ChromaDB vector store.
3. **Reason** — Claude generates a structured report (one card per defect plus an overall verdict) using only the retrieved context.
4. **Evaluate** — Three independent checks score the report; an overall pass flag is set only if all three pass.
5. **Export** — Results are returned as JSON and can be downloaded as a styled PDF.

```
        +--------+     +-------+     +-----------+     +--------+     +------------+
image → |  YOLO  | →   |  RAG  | →   |  Claude   | →   |  Eval  | →   |  JSON/PDF  |
        | (best) |     | Chroma|     | (Sonnet)  |     | x 3    |     |  response  |
        +--------+     +-------+     +-----------+     +--------+     +------------+
```

---

## Features

- **Six-class PCB defect detector** — fine-tuned YOLOv8 nano (`YOLO/VLDA/pcb_defect_nano_v1/weights/best.pt`).
- **IPC-A-610G grounding** — pre-built ChromaDB knowledge base shipped under [RAG/ipc_vectordb/](RAG/ipc_vectordb/).
- **Structured report generation** — Claude output parsed into per-defect cards (root cause, remediation, IPC reference, severity).
- **RAGAs-style metrics** — context precision/recall, faithfulness, answer relevancy.
- **LLM-Judge / G-Eval** — five quality dimensions scored 1–5 by Claude.
- **Citation verification** — every cited clause checked against the vector store and IPC acceptability vocabulary.
- **PDF export** — multi-page report via ReportLab Platypus.
- **Web UI** — single-page vanilla JS frontend at `/`.

---

## Project Structure

```
Vision-Language-Diagnostic-Agent-main/
├── app.py                      # Flask entry point (port 5000)
├── config.py                   # Centralised constants & thresholds
├── requirements.txt
│
├── models/
│   ├── yolo_model.py           # YOLOv8 inference wrapper
│   └── embed_model.py          # SentenceTransformer (BAAI BGE) wrapper
│
├── routes/
│   ├── analyze.py              # POST /analyze
│   ├── health.py               # GET  /health
│   └── pdf_report.py           # POST /download-report
│
├── services/
│   ├── rag_service.py          # ChromaDB retrieval
│   ├── report_service.py       # Claude prompt + parser
│   └── evaluation/
│       ├── ragas_eval.py       # Retrieval & generation metrics
│       ├── llm_judge.py        # G-Eval (5 dimensions)
│       └── citation_verify.py  # Semantic + lexical citation check
│
├── RAG/
│   ├── ipc_vector_pipeline.py  # PDF → chunks → ChromaDB ingestion
│   ├── requirements.txt
│   └── ipc_vectordb/           # Pre-built persistent Chroma collection
│
├── YOLO/
│   ├── data.yaml               # Dataset config (6 classes)
│   ├── preprocessing.ipynb
│   ├── train_yolo.ipynb
│   ├── test_app.py             # API smoke-test client
│   └── VLDA/pcb_defect_nano_v1/
│       └── weights/best.pt     # Fine-tuned weights
│
└── templates/
    └── index.html              # Web UI
```

---

## Tech Stack

| Layer            | Tool                                              |
|------------------|---------------------------------------------------|
| Web framework    | Flask                                             |
| Detection        | Ultralytics YOLOv8                                |
| Embeddings       | `BAAI/bge-base-en-v1.5` (sentence-transformers)   |
| Vector store     | ChromaDB (persistent)                             |
| Reasoning LLM    | Anthropic Claude (`claude-sonnet-4-20250514`)     |
| PDF              | ReportLab Platypus                                |
| Image I/O        | Pillow                                            |

---

## Defect Classes

The runtime label map is defined in [config.py:19-26](config.py#L19-L26):

| ID | Label             |
|----|-------------------|
| 0  | Open Circuit      |
| 1  | Short Circuit     |
| 2  | Mouse Bite        |
| 3  | Spur              |
| 4  | Spurious Copper   |
| 5  | Missing Hole      |

---

## Prerequisites

- **Python 3.10 or newer.**
- **An Anthropic API key** with access to `claude-sonnet-4-20250514`.
- The pre-built knowledge base at [RAG/ipc_vectordb/](RAG/ipc_vectordb/) (already in the repo).
- The YOLO weights at [YOLO/VLDA/pcb_defect_nano_v1/weights/best.pt](YOLO/VLDA/pcb_defect_nano_v1/weights/best.pt). The default path in [config.py](config.py) points to `YOLO/VLDA/yolov8n.pt`; either copy/rename the trained weights to that path or override `YOLO_MODEL_PATH` in your `.env`.

---

## Installation

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Create a .env file in the project root:
#   ANTHROPIC_API_KEY=sk-ant-...
#   YOLO_MODEL_PATH=YOLO/VLDA/pcb_defect_nano_v1/weights/best.pt
#   CHROMA_DIR=RAG/ipc_vectordb
```

---

## Running the Server

```bash
python app.py
```

The server starts on `http://0.0.0.0:5000`. Open `http://localhost:5000` in a browser for the upload UI.

---

## API Reference

### `GET /health`

Returns runtime metadata including loaded model paths and the number of chunks in the knowledge base.

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "ok",
  "yolo":   "YOLO/VLDA/pcb_defect_nano_v1/weights/best.pt",
  "embed":  "BAAI/bge-base-en-v1.5",
  "claude": "claude-sonnet-4-20250514",
  "ipc_chunks": 1234
}
```

### `POST /analyze`

Run the full pipeline on a single image. Accepts either multipart form-data or JSON.

**Form-data:**
```bash
curl -X POST http://localhost:5000/analyze -F "image=@pcb.jpg"
```

**JSON (base64):**
```bash
curl -X POST http://localhost:5000/analyze \
     -H "Content-Type: application/json" \
     -d "{\"image_base64\": \"<base64-encoded-image>\"}"
```

**Response shape** (see [routes/analyze.py](routes/analyze.py)):
```json
{
  "detections":    [{"class_id": 1, "label": "Short Circuit", "confidence": 0.91, "bbox": [...]}],
  "defect_cards":  [{"label": "...", "root_cause": "...", "remediation": "...", "ipc_reference": "..."}],
  "verdict":       "REJECT",
  "verdict_reason": "...",
  "evaluation": {
    "ragas":     { "...": "..." },
    "llm_judge": { "...": "..." },
    "citations": { "...": "..." },
    "overall_pass": false
  }
}
```

If no defect clears the confidence threshold, the endpoint short-circuits with `verdict: "ACCEPT"` and an empty evaluation block.

### `POST /download-report`

Pass the body returned by `/analyze` and receive a PDF.

```bash
curl -X POST http://localhost:5000/download-report \
     -H "Content-Type: application/json" \
     -d @analysis.json \
     -o report.pdf
```

---

## Web UI Usage

1. Visit `http://localhost:5000`.
2. Drag-and-drop or select a PCB image.
3. Click **Run Analysis** — annotated detections, per-defect cards, verdict, and evaluation scores render in the page.
4. Click **Download Report** to save a styled PDF of the same payload.

---

## Configuration

All runtime constants live in [config.py](config.py). The most useful are:

| Constant                       | Default                       | Purpose                                          |
|--------------------------------|-------------------------------|--------------------------------------------------|
| `YOLO_MODEL_PATH`              | `YOLO/VLDA/yolov8n.pt`        | Path to YOLO weights (env-overridable)           |
| `CHROMA_DIR`                   | `RAG/ipc_vectordb`            | Persistent Chroma directory (env-overridable)    |
| `EMBED_MODEL`                  | `BAAI/bge-base-en-v1.5`       | Sentence-transformer model                       |
| `CLAUDE_MODEL`                 | `claude-sonnet-4-20250514`    | Anthropic model id                               |
| `YOLO_CONFIDENCE`              | `0.25`                        | Minimum YOLO confidence to keep a detection      |
| `TOP_K_CHUNKS`                 | `4`                           | Chunks retrieved from IPC per defect             |
| `RAGAS_RETRIEVAL_PASS`         | `0.70`                        | Pass threshold for retrieval metrics             |
| `RAGAS_GENERATION_PASS`        | `0.70`                        | Pass threshold for generation metrics            |
| `CITATION_DISTANCE_THRESHOLD`  | `0.40`                        | Cosine distance below which a citation is a hit  |

Environment variables read at startup: `ANTHROPIC_API_KEY`, `YOLO_MODEL_PATH`, `CHROMA_DIR`.

---

## Evaluation Pipeline

Each call to `/analyze` runs three independent quality checks. `evaluation.overall_pass` is `true` only when every check passes.

### 1. RAGAs — [services/evaluation/ragas_eval.py](services/evaluation/ragas_eval.py)

- **Retrieval** — `context_precision` and `context_recall` computed from embedding similarity between defect queries and retrieved chunks.
- **Generation** — `faithfulness` (Claude scores how many report claims are grounded in retrieved context) and `answer_relevancy` (embedding similarity between query and report excerpt).
- **Pass rule** — average retrieval ≥ 0.70 **and** average generation ≥ 0.70.

### 2. LLM Judge / G-Eval — [services/evaluation/llm_judge.py](services/evaluation/llm_judge.py)

Claude scores the report on five dimensions (1–5):

- IPC Grounding
- Defect Coverage
- Root Cause Quality
- Remediation Quality
- Hallucination Risk

**Pass rule** — every dimension ≥ 3 and overall score ≥ 3.

### 3. Citation Verification — [services/evaluation/citation_verify.py](services/evaluation/citation_verify.py)

For each defect card:

- **Semantic match** — cited clause must score within cosine distance `CITATION_DISTANCE_THRESHOLD` (`0.40`) of the defect query.
- **Acceptability language** — the matched chunk must contain IPC vocabulary listed in [config.py](config.py) (`shall`, `class 1/2/3`, `target condition`, etc.).

A card is `PASS` (both checks), `PARTIAL` (semantic only), or `FAIL`. Aggregate `score = (PASS + 0.5·PARTIAL) / total`. **Pass rule** — score ≥ 0.80.

---

## Rebuilding the Knowledge Base

The repository ships a pre-built ChromaDB collection in [RAG/ipc_vectordb/](RAG/ipc_vectordb/). To rebuild against a new PDF:

```bash
# 1. Place your IPC-A-610G PDF in RAG/
# 2. Install RAG-pipeline-only deps (PyMuPDF, pytesseract, pdf2image)
pip install -r RAG/requirements.txt

# 3. Re-ingest
python RAG/ipc_vector_pipeline.py
```

---

## Training / Fine-tuning YOLO

The repository includes notebooks for the full training loop:

- [YOLO/preprocessing.ipynb](YOLO/preprocessing.ipynb) — dataset preparation.
- [YOLO/train_yolo.ipynb](YOLO/train_yolo.ipynb) — YOLOv8 fine-tuning.

The dataset layout expected by [YOLO/data.yaml](YOLO/data.yaml) is:

```
YOLO/yolo_dataset/
├── images/{train,val}/
└── labels/{train,val}/
```

The bundled fine-tuned model is based on the **DeepPCB** dataset.

---

## Acknowledgements

- **IPC-A-610G** — *Acceptability of Electronic Assemblies* (IPC, 2017).
- **DeepPCB** — public PCB defect detection dataset.
- **Ultralytics** — YOLOv8 implementation.
- **Anthropic** — Claude API.
- **BAAI** — `bge-base-en-v1.5` embedding model.
