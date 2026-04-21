"""
Evaluation 2 — LLM-as-Judge (G-Eval)

A second Claude call scores the generated report on 5 dimensions.
Each dimension returns: score (1–5), justification, pass/fail.
Overall PASS = all dimensions score ≥ 3.
"""

import json, re, anthropic
from config import CLAUDE_MODEL

PASS_THRESHOLD = 3   # out of 5

DIMENSIONS = [
    ("ipc_grounding",
     "IPC Grounding",
     "Does every defect classification and criterion cited trace back to a specific, "
     "real IPC-A-610G section number provided in the retrieved context? "
     "Vague references like 'per IPC standards' without a section number score 1–2. "
     "Exact section citations that match the retrieved context score 4–5."),

    ("defect_coverage",
     "Defect Coverage",
     "Is every defect detected by YOLO addressed in the report with its own entry? "
     "Missing any detected defect is a critical failure (score 1). "
     "All defects addressed with full detail scores 5."),

    ("root_cause_quality",
     "Root Cause Quality",
     "Are root causes technically specific to PCB manufacturing? "
     "'Manufacturing error' or 'process issue' are too generic (score 1–2). "
     "A good root cause names the specific process step, e.g. "
     "'stencil aperture over-printing during solder paste deposition' (score 4–5)."),

    ("remediation_quality",
     "Remediation Quality",
     "Are remediation steps actionable, ordered correctly, and consistent with "
     "IPC-7711/7721 rework practices? Steps that simply say 'repair the defect' score 1. "
     "Steps that specify tools, techniques, and verification checks score 4–5."),

    ("hallucination_risk",
     "Hallucination Risk",
     "Does the report invent IPC section numbers, criteria thresholds, or acceptability "
     "conditions NOT present in the provided context? "
     "Score 5 = no fabrication found. Score 1 = clear fabrication of standard references."),
]


def run(raw_report: str, detections: list[dict], rag_contexts: dict) -> dict:
    context_block = "\n".join(
        f"[{defect} | Page {c['page']}]: {c['text'][:300]}"
        for defect, chunks in rag_contexts.items()
        for c in chunks
    )
    dim_text = "\n".join(
        f'{i+1}. "{key}" ({label}): {desc}'
        for i, (key, label, desc) in enumerate(DIMENSIONS)
    )

    prompt = f"""You are a senior IPC-A-610G certified quality engineer evaluating an AI-generated PCB inspection report.

--- YOLO DETECTIONS (ground truth) ---
{json.dumps(detections, indent=2)}

--- RETRIEVED IPC-A-610G CONTEXT ---
{context_block}

--- AI-GENERATED REPORT ---
{raw_report}

Score the report on these 5 dimensions (1 = very poor, 5 = excellent):
{dim_text}

Respond ONLY with valid JSON, no markdown fences, exactly this schema:
{{
  "ipc_grounding":       {{"score": <1-5>, "justification": "<2-3 sentences referencing specific evidence>"}},
  "defect_coverage":     {{"score": <1-5>, "justification": "<2-3 sentences>"}},
  "root_cause_quality":  {{"score": <1-5>, "justification": "<2-3 sentences>"}},
  "remediation_quality": {{"score": <1-5>, "justification": "<2-3 sentences>"}},
  "hallucination_risk":  {{"score": <1-5>, "justification": "<2-3 sentences>"}}
}}"""

    try:
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = re.sub(r"^```[a-z]*\n?", "", msg.content[0].text.strip()).rstrip("`").strip()
        data = json.loads(raw)

        # Attach pass/fail per dimension
        for key, *_ in DIMENSIONS:
            if key in data and isinstance(data[key], dict):
                data[key]["pass"]  = data[key].get("score", 0) >= PASS_THRESHOLD
                data[key]["label"] = "PASS" if data[key]["pass"] else "FAIL"

        scores = [data[k]["score"] for k, *_ in DIMENSIONS if isinstance(data.get(k), dict)]
        overall_score = round(sum(scores) / len(scores), 2) if scores else 0
        overall_pass  = all(data.get(k, {}).get("pass", False) for k, *_ in DIMENSIONS)

        data["overall"] = {
            "score":    overall_score,
            "pass":     overall_pass,
            "label":    "PASS" if overall_pass else "FAIL",
            "summary":  _summary(data),
        }
        data["error"] = None
        return data

    except Exception as e:
        return _empty(str(e))


def _summary(data: dict) -> str:
    failed = [label for key, label, _ in DIMENSIONS if not data.get(key, {}).get("pass", True)]
    if not failed:
        return "Report meets quality standards on all five dimensions."
    return f"Failed dimensions require attention: {', '.join(failed)}."


def _empty(error: str) -> dict:
    base = {"score": 0, "justification": "Evaluation failed.", "pass": False, "label": "FAIL"}
    result = {k: dict(base) for k, *_ in DIMENSIONS}
    result["overall"] = {"score": 0, "pass": False, "label": "FAIL",
                         "summary": "Evaluation could not be completed."}
    result["error"] = error
    return result
