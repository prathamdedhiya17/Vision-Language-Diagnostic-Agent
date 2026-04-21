import re, json, anthropic
from config import CLAUDE_MODEL

_client = None

def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

def generate(detections: list[dict], rag_contexts: dict) -> str:
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

For EACH defect use EXACTLY this format (keep labels verbatim):

DEFECT_START
Defect Type: <name>
Defect ID: <DFT-001, DFT-002 ...>
Location: <x1,y1 to x2,y2>
Confidence: <percentage>
IPC Classification: <Target / Acceptable / Defect — Class 1/2/3>
IPC Reference: <exact section number and title from the context above>
Severity: <Critical / Major / Minor>
Root Cause: <one technically specific sentence>
Remediation: <numbered steps, one per line>
DEFECT_END

After all defects:
VERDICT: <ACCEPT | REJECT | REWORK REQUIRED>
VERDICT_REASON: <one sentence citing which defect drove the verdict>
"""
    msg = get_client().messages.create(
        model=CLAUDE_MODEL, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def parse(raw: str, detections: list[dict]) -> dict:
    cards   = []
    verdict = "UNKNOWN"
    reason  = ""

    vm = re.search(r"VERDICT:\s*(.+)",        raw)
    rm = re.search(r"VERDICT_REASON:\s*(.+)", raw)
    if vm: verdict = vm.group(1).strip().upper()
    if rm: reason  = rm.group(1).strip()

    for i, block in enumerate(re.findall(r"DEFECT_START(.*?)DEFECT_END", raw, re.DOTALL)):
        def field(label):
            m = re.search(rf"{label}:\s*(.+)", block)
            return m.group(1).strip() if m else "—"

        rem_raw   = field("Remediation")
        rem_steps = [s.strip() for s in re.split(r"\d+\.\s+", rem_raw) if s.strip()] or [rem_raw]

        conf_str = field("Confidence").replace("%", "").strip()
        try:    conf = float(conf_str) / 100
        except: conf = detections[i]["confidence"] if i < len(detections) else 0

        cards.append({
            "defect_id":          field("Defect ID"),
            "defect_type":        field("Defect Type"),
            "location":           field("Location"),
            "confidence":         conf,
            "ipc_classification": field("IPC Classification"),
            "ipc_reference":      field("IPC Reference"),
            "severity":           field("Severity"),
            "root_cause":         field("Root Cause"),
            "remediation":        rem_steps,
        })

    return {"verdict": verdict, "verdict_reason": reason, "defect_cards": cards, "raw_report": raw}
