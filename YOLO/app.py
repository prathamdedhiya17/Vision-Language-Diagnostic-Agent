"""
VLDA — Vision-Language Diagnostic Agent
Flask app with embedded single-page frontend

pip install flask ultralytics chromadb sentence-transformers anthropic pillow python-dotenv
"""

import io, json, base64, re, uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import anthropic
import chromadb
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO

# ── CONFIG ───────────────────────────────────────────────
YOLO_MODEL_PATH = "VLDA/yolov8n.pt"
CHROMA_DIR      = "../RAG/ipc_vectordb"
COLLECTION      = "ipc_a_610g"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
CLAUDE_MODEL    = "claude-sonnet-4-20250514"
TOP_K_CHUNKS    = 4
CONFIDENCE      = 0.25

DEFECT_LABELS = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mousebite",
    3: "spur",
    4: "pin_hole",
    5: "spurious_copper"
}

app = Flask(__name__)

print("Loading models...")
yolo_model    = YOLO(YOLO_MODEL_PATH)
embed_model   = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection    = chroma_client.get_collection(COLLECTION)
claude        = anthropic.Anthropic()
print("✅ Ready.\n")


# ── BACKEND LOGIC ────────────────────────────────────────
def run_yolo(image):
    results  = yolo_model.predict(image, conf=CONFIDENCE, verbose=False)[0]
    detected = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detected.append({
            "defect_type": DEFECT_LABELS.get(cls_id, f"Class {cls_id}"),
            "confidence":  round(float(box.conf[0]), 3),
            "bbox":        [round(v, 1) for v in box.xyxy[0].tolist()]
        })
    return detected

def retrieve_ipc_context(defect_type):
    results = collection.query(
        query_embeddings = embed_model.encode([defect_type]).tolist(),
        n_results        = TOP_K_CHUNKS,
        include          = ["documents", "metadatas", "distances"]
    )
    return [
        {"text": doc, "page": meta.get("page","?"), "section": meta.get("section","general"),
         "relevance": round(1 - dist, 3)}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

def generate_report(detections, rag_contexts):
    context_block = ""
    for defect, chunks in rag_contexts.items():
        context_block += f"\n### IPC-A-610G — {defect}\n"
        for c in chunks:
            context_block += f"[Page {c['page']} | {c['section']}]: {c['text']}\n\n"

    prompt = f"""You are a PCB quality assurance engineer writing a formal inspection report.
YOLO detected these defects:
{json.dumps(detections, indent=2)}

IPC-A-610G References:
{context_block}

Write a structured report. For EACH defect use EXACTLY this format (keep the labels verbatim):

DEFECT_START
Defect Type: <name>
Defect ID: <DFT-001, DFT-002, ...>
Location: <x1,y1 to x2,y2 from bbox>
Confidence: <percentage>
IPC Classification: <Target / Acceptable / Defect — Class 1/2/3>
IPC Reference: <section number and title>
Severity: <Critical / Major / Minor>
Root Cause: <one clear sentence explaining why this occurs>
Remediation: <numbered steps, one per line>
DEFECT_END

After all defects, write:
VERDICT: <ACCEPT | REJECT | REWORK REQUIRED>
VERDICT_REASON: <one sentence justification>
"""
    message = claude.messages.create(
        model=CLAUDE_MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ── HTML TEMPLATE ────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLDA — PCB Inspection System</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Barlow:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #07090c;
    --surface:   #0e1218;
    --border:    #1c2330;
    --border2:   #263040;
    --green:     #00e5a0;
    --green-dim: #00e5a015;
    --amber:     #f5a623;
    --red:       #ff4d4d;
    --blue:      #4d9eff;
    --text:      #c8d6e8;
    --muted:     #5a6a7e;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Barlow', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    background-position: -1px -1px;
  }

  /* ── HEADER ── */
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border2);
    padding: 0 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    height: 56px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .logo-dot {
    width: 10px; height: 10px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .logo-text {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 2px;
  }
  .logo-sub {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    margin-left: 4px;
  }
  .header-right {
    margin-left: auto;
    display: flex;
    gap: 24px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
  .header-right span b { color: var(--text); }

  /* ── MAIN LAYOUT ── */
  main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
    display: grid;
    grid-template-columns: 380px 1fr;
    gap: 24px;
    align-items: start;
  }

  /* ── PANEL ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 4px;
    overflow: hidden;
  }
  .panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }
  .panel-header .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--border2);
  }
  .panel-header .dot.active { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .panel-body { padding: 20px; }

  /* ── UPLOAD ZONE ── */
  #drop-zone {
    border: 1px dashed var(--border2);
    border-radius: 4px;
    padding: 32px 16px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    position: relative;
  }
  #drop-zone:hover, #drop-zone.drag-over {
    border-color: var(--green);
    background: var(--green-dim);
  }
  #drop-zone input { display: none; }
  .upload-icon {
    font-size: 28px;
    margin-bottom: 12px;
    opacity: .5;
  }
  .upload-label {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.6;
  }
  .upload-label b { color: var(--green); font-weight: 600; }
  .upload-hint {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 8px;
  }

  /* ── IMAGE PREVIEW ── */
  #preview-wrap { display: none; margin-bottom: 16px; }
  #preview-wrap img {
    width: 100%;
    border-radius: 4px;
    border: 1px solid var(--border2);
    display: block;
  }
  .preview-name {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
    padding: 4px 8px;
    background: var(--bg);
    border-radius: 3px;
  }

  /* ── BUTTON ── */
  #analyze-btn {
    width: 100%;
    margin-top: 16px;
    padding: 12px;
    background: var(--green);
    color: #000;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    border: none;
    border-radius: 3px;
    cursor: pointer;
    transition: all .2s;
    text-transform: uppercase;
  }
  #analyze-btn:hover:not(:disabled) {
    background: #00ffb3;
    box-shadow: 0 0 20px #00e5a040;
  }
  #analyze-btn:disabled {
    opacity: .4;
    cursor: not-allowed;
  }

  /* ── PROGRESS ── */
  #progress { display: none; margin-top: 16px; }
  .progress-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }
  .progress-step:last-child { border-bottom: none; }
  .step-icon { width: 16px; text-align: center; font-size: 13px; }
  .progress-step.done { color: var(--green); }
  .progress-step.active { color: var(--amber); }
  .spinner {
    width: 12px; height: 12px;
    border: 2px solid var(--border2);
    border-top-color: var(--amber);
    border-radius: 50%;
    animation: spin .6s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── REPORT PANEL ── */
  #report-panel { display: none; grid-column: 2; }

  /* ── REPORT META BAR ── */
  .report-meta {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }
  .meta-cell {
    background: var(--surface);
    padding: 12px 16px;
  }
  .meta-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .meta-value {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }

  /* ── VERDICT BANNER ── */
  #verdict-banner {
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--border);
  }
  #verdict-banner.accept  { background: #00e5a012; border-left: 4px solid var(--green); }
  #verdict-banner.reject  { background: #ff4d4d12; border-left: 4px solid var(--red); }
  #verdict-banner.rework  { background: #f5a62312; border-left: 4px solid var(--amber); }
  .verdict-badge {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 4px 10px;
    border-radius: 3px;
    text-transform: uppercase;
  }
  .accept  .verdict-badge { background: var(--green); color: #000; }
  .reject  .verdict-badge { background: var(--red);   color: #fff; }
  .rework  .verdict-badge { background: var(--amber); color: #000; }
  .verdict-reason {
    font-size: 13px;
    color: var(--text);
    font-weight: 300;
  }

  /* ── DEFECT CARDS ── */
  .defects-grid { padding: 20px; display: flex; flex-direction: column; gap: 16px; }

  .defect-card {
    border: 1px solid var(--border2);
    border-radius: 4px;
    overflow: hidden;
    animation: slideIn .3s ease;
  }
  @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

  .defect-card-header {
    padding: 10px 16px;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .defect-id {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }
  .defect-type-badge {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    background: var(--border2);
    color: var(--text);
    letter-spacing: .5px;
  }
  .severity-badge {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: auto;
  }
  .severity-badge.Critical { background:#ff4d4d20; color:var(--red); border:1px solid var(--red); }
  .severity-badge.Major    { background:#f5a62320; color:var(--amber); border:1px solid var(--amber); }
  .severity-badge.Minor    { background:#4d9eff20; color:var(--blue); border:1px solid var(--blue); }

  .defect-card-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
  }
  .field {
    padding: 10px 16px;
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
  }
  .field:nth-child(even) { border-right: none; }
  .field.full { grid-column: 1 / -1; border-right: none; }
  .field:last-child, .field:nth-last-child(2):not(.full) { border-bottom: none; }
  .field-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .field-value {
    font-size: 12px;
    color: var(--text);
    line-height: 1.5;
  }
  .field-value.mono {
    font-family: var(--mono);
    font-size: 11px;
  }
  .ipc-chip {
    display: inline-block;
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 2px;
    margin-top: 2px;
  }
  .ipc-chip.defect     { background:#ff4d4d15; color:var(--red);   border:1px solid #ff4d4d40; }
  .ipc-chip.acceptable { background:#f5a62315; color:var(--amber); border:1px solid #f5a62340; }
  .ipc-chip.target     { background:#00e5a015; color:var(--green); border:1px solid #00e5a040; }

  .conf-bar {
    height: 4px;
    background: var(--border2);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .conf-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--green);
    transition: width 1s ease;
  }

  .remediation-list {
    padding-left: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .remediation-list li {
    display: flex;
    gap: 8px;
    font-size: 12px;
    color: var(--text);
    line-height: 1.5;
  }
  .remediation-list li::before {
    content: counter(step);
    counter-increment: step;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: 2px;
    width: 18px;
    height: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
  }
  .remediation-list { counter-reset: step; }

  /* ── NO DEFECTS ── */
  .no-defects {
    padding: 48px 20px;
    text-align: center;
  }
  .no-defects .icon { font-size: 40px; margin-bottom: 12px; }
  .no-defects h3 { color: var(--green); font-size: 16px; margin-bottom: 6px; }
  .no-defects p  { color: var(--muted); font-size: 12px; font-family: var(--mono); }

  /* ── SUMMARY ROW ── */
  .defects-summary {
    display: flex;
    gap: 12px;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .sum-chip {
    font-family: var(--mono);
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid var(--border2);
    color: var(--muted);
  }
  .sum-chip b { color: var(--text); }

  @media(max-width: 900px) {
    main { grid-template-columns: 1fr; }
    #report-panel { grid-column: 1; }
    .report-meta { grid-template-columns: 1fr 1fr; }
    .defect-card-body { grid-template-columns: 1fr; }
    .field { border-right: none; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-dot"></div>
  <span class="logo-text">VLDA</span>
  <span class="logo-sub">Vision-Language Diagnostic Agent</span>
  <div class="header-right">
    <span>STANDARD <b>IPC-A-610G</b></span>
    <span>ENGINE <b>YOLOv8 + RAG</b></span>
    <span>MODEL <b>Claude Sonnet</b></span>
  </div>
</header>

<main>

  <!-- LEFT: Upload Panel -->
  <div>
    <div class="panel">
      <div class="panel-header">
        <div class="dot active"></div>
        PCB Image Input
      </div>
      <div class="panel-body">

        <div id="preview-wrap">
          <img id="preview-img" src="" alt="PCB Preview">
          <div class="preview-name" id="preview-name"></div>
        </div>

        <div id="drop-zone">
          <input type="file" id="file-input" accept="image/*">
          <div class="upload-icon">⬡</div>
          <div class="upload-label">
            <b>Click to upload</b> or drag & drop<br>
            a PCB board image
          </div>
          <div class="upload-hint">JPG · PNG · BMP · TIFF</div>
        </div>

        <button id="analyze-btn" disabled>RUN INSPECTION</button>

        <div id="progress">
          <div class="progress-step" id="step-yolo">
            <span class="step-icon">◈</span>
            <span>YOLO defect detection</span>
          </div>
          <div class="progress-step" id="step-rag">
            <span class="step-icon">◈</span>
            <span>IPC-A-610G retrieval (RAG)</span>
          </div>
          <div class="progress-step" id="step-claude">
            <span class="step-icon">◈</span>
            <span>Claude report generation</span>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- RIGHT: Report Panel -->
  <div id="report-panel" class="panel">
    <div class="panel-header">
      <div class="dot active"></div>
      Inspection Report
      <span style="margin-left:auto;font-family:var(--mono);font-size:10px" id="report-id"></span>
    </div>

    <div class="report-meta">
      <div class="meta-cell">
        <div class="meta-label">Report ID</div>
        <div class="meta-value mono" id="meta-rid">—</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Timestamp</div>
        <div class="meta-value mono" id="meta-ts">—</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Defects Found</div>
        <div class="meta-value mono" id="meta-count">—</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Standard</div>
        <div class="meta-value mono">IPC-A-610G</div>
      </div>
    </div>

    <div id="verdict-banner">
      <span class="verdict-badge" id="verdict-text">—</span>
      <span class="verdict-reason" id="verdict-reason">—</span>
    </div>

    <div class="defects-summary" id="defects-summary"></div>
    <div class="defects-grid" id="defects-grid"></div>
  </div>

</main>

<script>
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const progress    = document.getElementById('progress');
const reportPanel = document.getElementById('report-panel');

let selectedFile = null;

// ── File selection ──
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => setFile(e.target.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  setFile(e.dataTransfer.files[0]);
});

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.style.display = 'block';
    previewName.textContent = file.name + ' · ' + (file.size / 1024).toFixed(1) + ' KB';
    dropZone.style.display = 'none';
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  reportPanel.style.display = 'none';
}

// ── Progress helpers ──
function setStep(id, state) {
  const el = document.getElementById(id);
  el.className = 'progress-step ' + state;
  if (state === 'active') {
    el.querySelector('.step-icon').innerHTML = '<span class="spinner"></span>';
  } else if (state === 'done') {
    el.querySelector('.step-icon').textContent = '✓';
  }
}

// ── Analyze ──
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  progress.style.display = 'block';
  reportPanel.style.display = 'none';
  ['step-yolo','step-rag','step-claude'].forEach(s => {
    const el = document.getElementById(s);
    el.className = 'progress-step';
    el.querySelector('.step-icon').textContent = '◈';
  });

  setStep('step-yolo', 'active');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: formData });
    setStep('step-yolo', 'done');
    setStep('step-rag', 'active');

    // slight visual delay so steps feel sequential
    await new Promise(r => setTimeout(r, 400));
    setStep('step-rag', 'done');
    setStep('step-claude', 'active');

    await new Promise(r => setTimeout(r, 300));
    const data = await res.json();
    setStep('step-claude', 'done');

    renderReport(data);
  } catch(err) {
    alert('Error: ' + err.message);
  }

  analyzeBtn.disabled = false;
});

// ── Render report ──
function renderReport(data) {
  const rid = 'RPT-' + Math.random().toString(36).slice(2,8).toUpperCase();
  const ts  = new Date().toISOString().replace('T',' ').slice(0,19);

  document.getElementById('meta-rid').textContent   = rid;
  document.getElementById('meta-ts').textContent    = ts;
  document.getElementById('meta-count').textContent = data.detections.length;

  // Verdict
  const verdict = data.verdict || 'UNKNOWN';
  const vBanner = document.getElementById('verdict-banner');
  const vClass  = verdict.includes('REJECT') ? 'reject'
                : verdict.includes('REWORK') ? 'rework' : 'accept';
  vBanner.className = vClass;
  document.getElementById('verdict-text').textContent   = verdict;
  document.getElementById('verdict-reason').textContent = data.verdict_reason || '';

  // Summary chips
  const summaryEl = document.getElementById('defects-summary');
  const typeCounts = {};
  data.detections.forEach(d => typeCounts[d.defect_type] = (typeCounts[d.defect_type]||0)+1);
  summaryEl.innerHTML = Object.entries(typeCounts)
    .map(([t,c]) => `<div class="sum-chip"><b>${c}</b> ${t}</div>`).join('');

  // Defect cards
  const grid = document.getElementById('defects-grid');
  grid.innerHTML = '';

  if (!data.defect_cards || data.defect_cards.length === 0) {
    grid.innerHTML = `<div class="no-defects">
      <div class="icon">✓</div>
      <h3>No Defects Detected</h3>
      <p>Board passed all IPC-A-610G acceptance criteria.</p>
    </div>`;
  } else {
    data.defect_cards.forEach((card, i) => {
      const conf = Math.round((card.confidence||0) * 100);
      const ipcClass = (card.ipc_classification||'').toLowerCase().includes('defect') ? 'defect'
                     : (card.ipc_classification||'').toLowerCase().includes('accept') ? 'acceptable' : 'target';
      const sevClass = card.severity || 'Minor';

      const remItems = (card.remediation || [])
        .map(r => `<li>${r}</li>`).join('');

      grid.innerHTML += `
      <div class="defect-card" style="animation-delay:${i*80}ms">
        <div class="defect-card-header">
          <span class="defect-id">${card.defect_id || `DFT-${String(i+1).padStart(3,'0')}`}</span>
          <span class="defect-type-badge">${card.defect_type}</span>
          <span class="severity-badge ${sevClass}">${sevClass}</span>
        </div>
        <div class="defect-card-body">
          <div class="field">
            <div class="field-label">Location (bbox)</div>
            <div class="field-value mono">${card.location || '—'}</div>
          </div>
          <div class="field">
            <div class="field-label">Detection Confidence</div>
            <div class="field-value mono">${conf}%</div>
            <div class="conf-bar"><div class="conf-fill" style="width:${conf}%;background:${conf>80?'var(--green)':conf>50?'var(--amber)':'var(--red)'}"></div></div>
          </div>
          <div class="field">
            <div class="field-label">IPC-A-610G Classification</div>
            <div class="field-value">
              <span class="ipc-chip ${ipcClass}">${card.ipc_classification || '—'}</span>
            </div>
          </div>
          <div class="field">
            <div class="field-label">IPC Reference</div>
            <div class="field-value mono">${card.ipc_reference || '—'}</div>
          </div>
          <div class="field full">
            <div class="field-label">Root Cause</div>
            <div class="field-value">${card.root_cause || '—'}</div>
          </div>
          <div class="field full">
            <div class="field-label">Recommended Remediation</div>
            <div class="field-value">
              <ul class="remediation-list">${remItems}</ul>
            </div>
          </div>
        </div>
      </div>`;
    });
  }

  reportPanel.style.display = 'block';
  reportPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>
</body>
</html>"""


# ── PARSE CLAUDE REPORT INTO STRUCTURED CARDS ────────────
def parse_report(raw: str, detections: list) -> dict:
    """Extract structured defect cards and verdict from Claude's text output."""
    cards   = []
    verdict = "UNKNOWN"
    reason  = ""

    # Extract verdict
    vm = re.search(r"VERDICT:\s*(.+)", raw)
    rm = re.search(r"VERDICT_REASON:\s*(.+)", raw)
    if vm: verdict = vm.group(1).strip().upper()
    if rm: reason  = rm.group(1).strip()

    # Extract per-defect blocks
    blocks = re.findall(r"DEFECT_START(.*?)DEFECT_END", raw, re.DOTALL)
    for i, block in enumerate(blocks):
        def get_field(label):
            m = re.search(rf"{label}:\s*(.+)", block)
            return m.group(1).strip() if m else "—"

        # Parse remediation as numbered list
        rem_raw = get_field("Remediation")
        rem_steps = re.split(r"\d+\.\s+", rem_raw)
        rem_steps = [s.strip() for s in rem_steps if s.strip()]
        if not rem_steps:
            rem_steps = [rem_raw]

        conf_str = get_field("Confidence").replace("%","").strip()
        try:    conf = float(conf_str) / 100
        except: conf = detections[i]["confidence"] if i < len(detections) else 0

        cards.append({
            "defect_id":          get_field("Defect ID"),
            "defect_type":        get_field("Defect Type"),
            "location":           get_field("Location"),
            "confidence":         conf,
            "ipc_classification": get_field("IPC Classification"),
            "ipc_reference":      get_field("IPC Reference"),
            "severity":           get_field("Severity"),
            "root_cause":         get_field("Root Cause"),
            "remediation":        rem_steps,
        })

    return {"verdict": verdict, "verdict_reason": reason, "defect_cards": cards}


# ── ROUTES ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")
    elif request.is_json and "image_base64" in request.json:
        image = Image.open(io.BytesIO(base64.b64decode(request.json["image_base64"]))).convert("RGB")
    else:
        return jsonify({"error": "No image provided."}), 400

    detections = run_yolo(image)

    if not detections:
        return jsonify({
            "detections":    [],
            "defect_cards":  [],
            "verdict":       "ACCEPT",
            "verdict_reason": "No defects detected above confidence threshold.",
        })

    rag_contexts = {}
    for det in detections:
        if det["defect_type"] not in rag_contexts:
            rag_contexts[det["defect_type"]] = retrieve_ipc_context(det["defect_type"])

    raw_report = generate_report(detections, rag_contexts)
    parsed     = parse_report(raw_report, detections)

    return jsonify({"detections": detections, **parsed})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "chunks": collection.count()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)