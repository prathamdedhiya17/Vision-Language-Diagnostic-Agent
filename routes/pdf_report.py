"""
routes/pdf_report.py
POST /download-report
Accepts the full JSON response from /analyze and returns a formatted PDF.
Uses reportlab Platypus for structured, multi-page report layout.

pip install reportlab
"""

import io
from datetime import datetime
from flask import Blueprint, request, send_file, jsonify

from reportlab.lib.pagesizes   import A4
from reportlab.lib.units        import mm
from reportlab.lib              import colors
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums        import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus         import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)

pdf_bp = Blueprint("pdf_report", __name__)

# ── Colours ──────────────────────────────────────────────
C_BLACK   = colors.HexColor("#0a0a0a")
C_DARK    = colors.HexColor("#1a1a2e")
C_GREEN   = colors.HexColor("#00b87a")
C_AMBER   = colors.HexColor("#d4882a")
C_RED     = colors.HexColor("#d93025")
C_BLUE    = colors.HexColor("#2a6dd9")
C_LGRAY   = colors.HexColor("#f4f5f7")
C_MGRAY   = colors.HexColor("#8a9ab0")
C_BGRAY   = colors.HexColor("#dde1e8")
C_WHITE   = colors.white


def severity_color(sev: str):
    return {"Critical": C_RED, "Major": C_AMBER, "Minor": C_BLUE}.get(sev, C_BLUE)

def verdict_color(v: str):
    v = v.upper()
    if "REJECT" in v: return C_RED
    if "REWORK" in v: return C_AMBER
    return C_GREEN

def eval_color(passed: bool):
    return C_GREEN if passed else C_RED


# ── Style sheet ───────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "cover_title":   s("ct",  fontSize=26, textColor=C_WHITE,   fontName="Helvetica-Bold",   leading=32, alignment=TA_LEFT),
        "cover_sub":     s("cs",  fontSize=11, textColor=C_BGRAY,   fontName="Helvetica",        leading=16, alignment=TA_LEFT),
        "cover_meta":    s("cm",  fontSize=9,  textColor=C_BGRAY,   fontName="Helvetica",        leading=14, alignment=TA_LEFT),
        "section_hdr":   s("sh",  fontSize=10, textColor=C_WHITE,   fontName="Helvetica-Bold",   leading=14, alignment=TA_LEFT),
        "field_label":   s("fl",  fontSize=7,  textColor=C_MGRAY,   fontName="Helvetica-Bold",   leading=10, spaceAfter=1, spaceBefore=2),
        "field_value":   s("fv",  fontSize=9,  textColor=C_BLACK,   fontName="Helvetica",        leading=13),
        "field_mono":    s("fm",  fontSize=8,  textColor=C_BLACK,   fontName="Courier",          leading=12),
        "body":          s("b",   fontSize=9,  textColor=C_BLACK,   fontName="Helvetica",        leading=14),
        "body_bold":     s("bb",  fontSize=9,  textColor=C_BLACK,   fontName="Helvetica-Bold",   leading=14),
        "rem_item":      s("ri",  fontSize=8,  textColor=C_BLACK,   fontName="Helvetica",        leading=12, leftIndent=10),
        "eval_label":    s("el",  fontSize=7,  textColor=C_MGRAY,   fontName="Helvetica-Bold",   leading=10, spaceAfter=1),
        "eval_val":      s("ev",  fontSize=9,  textColor=C_BLACK,   fontName="Helvetica-Bold",   leading=12),
        "just_text":     s("jt",  fontSize=8,  textColor=colors.HexColor("#444"), fontName="Helvetica", leading=12),
        "footer":        s("ft",  fontSize=7,  textColor=C_MGRAY,   fontName="Helvetica",        leading=10, alignment=TA_CENTER),
        "toc_item":      s("ti",  fontSize=9,  textColor=C_BLACK,   fontName="Helvetica",        leading=14),
    }


# ── Page template (header/footer) ─────────────────────────
class ReportCanvas:
    """Mixin applied via onFirstPage / onLaterPages callbacks."""

    @staticmethod
    def draw_header(canvas, doc, rid, ts):
        canvas.saveState()
        w, h = A4
        # top bar
        canvas.setFillColor(C_DARK)
        canvas.rect(0, h - 14*mm, w, 14*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(C_WHITE)
        canvas.drawString(15*mm, h - 9*mm, "VLDA  ·  PCB INSPECTION REPORT")
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_MGRAY)
        canvas.drawRightString(w - 15*mm, h - 9*mm, f"{rid}  ·  {ts}")
        canvas.restoreState()

    @staticmethod
    def draw_footer(canvas, doc):
        canvas.saveState()
        w, _ = A4
        canvas.setStrokeColor(C_BGRAY)
        canvas.setLineWidth(0.5)
        canvas.line(15*mm, 12*mm, w - 15*mm, 12*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(C_MGRAY)
        canvas.drawString(15*mm,  8*mm, "IPC-A-610G  ·  Automated Inspection  ·  VLDA System")
        canvas.drawRightString(w - 15*mm, 8*mm, f"Page {doc.page}")
        canvas.restoreState()


# ── PDF builder ───────────────────────────────────────────
def build_pdf(data: dict) -> bytes:
    buf    = io.BytesIO()
    W, H   = A4
    M      = 15*mm
    styles = make_styles()

    rid = data.get("report_id", f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    ts  = data.get("timestamp",  datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    verdict     = data.get("verdict",       "UNKNOWN")
    vr          = data.get("verdict_reason","—")
    detections  = data.get("detections",    [])
    cards       = data.get("defect_cards",  [])
    evaluation  = data.get("evaluation",    {})
    overall_pass = data.get("overall_pass", False)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=20*mm, bottomMargin=18*mm,
        title=f"VLDA Inspection Report {rid}",
        author="VLDA System",
    )

    def on_page(canvas, doc):
        ReportCanvas.draw_header(canvas, doc, rid, ts)
        ReportCanvas.draw_footer(canvas, doc)

    story = []

    # ── COVER ──────────────────────────────────────────────
    story += _cover(styles, rid, ts, verdict, vr, len(cards), overall_pass)
    story.append(PageBreak())

    # ── DEFECT CARDS ───────────────────────────────────────
    story += [_section_header(styles, "DEFECT FINDINGS")]
    if not cards:
        story.append(Paragraph("No defects detected above confidence threshold.", styles["body"]))
    else:
        for i, card in enumerate(cards):
            story += _defect_card(styles, card, i)
            story.append(Spacer(1, 6*mm))

    story.append(PageBreak())

    # ── EVALUATION ─────────────────────────────────────────
    story += _evaluation_section(styles, evaluation, overall_pass)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf.read()


# ── Cover page ────────────────────────────────────────────
def _cover(styles, rid, ts, verdict, vr, n_defects, overall_pass):
    W, H = A4
    elems = []

    # Dark cover block (drawn as a table background)
    vc  = verdict_color(verdict)
    epc = C_GREEN if overall_pass else C_RED

    cover_data = [[
        Paragraph("VLDA", styles["cover_title"]),
    ]]
    cover_tbl = Table(cover_data, colWidths=[W - 30*mm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), C_DARK),
        ("TOPPADDING",  (0,0), (-1,-1), 10*mm),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4*mm),
        ("LEFTPADDING", (0,0), (-1,-1), 8*mm),
    ]))
    elems.append(cover_tbl)

    sub_data = [[Paragraph("Vision-Language Diagnostic Agent<br/>PCB Inspection Report", styles["cover_sub"])]]
    sub_tbl  = Table(sub_data, colWidths=[W - 30*mm])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",   (0,0),(-1,-1), 0),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8*mm),
        ("LEFTPADDING",  (0,0),(-1,-1), 8*mm),
    ]))
    elems.append(sub_tbl)
    elems.append(Spacer(1, 8*mm))

    # Meta grid
    meta_rows = [
        ["Report ID",       rid,           "Standard",     "IPC-A-610G"],
        ["Timestamp",       ts,            "Defects Found", str(n_defects)],
        ["Eval Framework",  "RAGAs + LLM Judge + Citation Verify", "Engine", "YOLOv8 + RAG + Claude"],
    ]
    meta_tbl_data = []
    for row in meta_rows:
        meta_tbl_data.append([
            Paragraph(row[0], styles["field_label"]),
            Paragraph(row[1], styles["field_mono"]),
            Paragraph(row[2], styles["field_label"]),
            Paragraph(row[3], styles["field_mono"]),
        ])
    meta_tbl = Table(meta_tbl_data, colWidths=[35*mm, 65*mm, 35*mm, 45*mm])
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_LGRAY),
        ("GRID",         (0,0),(-1,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 6),
    ]))
    elems.append(meta_tbl)
    elems.append(Spacer(1, 6*mm))

    # Verdict + eval pass strip
    vt_data = [[
        Paragraph(f"VERDICT: {verdict}", ParagraphStyle(
            "vt", fontSize=13, fontName="Helvetica-Bold",
            textColor=C_WHITE, leading=16)),
        Paragraph(vr, ParagraphStyle(
            "vr", fontSize=8, fontName="Helvetica",
            textColor=C_WHITE, leading=12)),
    ]]
    vt_tbl = Table(vt_data, colWidths=[55*mm, 125*mm])
    vt_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), vc),
        ("TOPPADDING",   (0,0),(-1,-1), 5*mm),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5*mm),
        ("LEFTPADDING",  (0,0),(-1,-1), 6*mm),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    elems.append(vt_tbl)
    elems.append(Spacer(1, 2*mm))

    ep_data = [[
        Paragraph("EVALUATION OUTCOME", ParagraphStyle(
            "ep", fontSize=8, fontName="Helvetica-Bold",
            textColor=C_WHITE, leading=10)),
        Paragraph("✓ ALL EVALUATIONS PASSED" if overall_pass else "✗ REVIEW REQUIRED — SEE EVALUATION SECTION",
            ParagraphStyle("epl", fontSize=9, fontName="Helvetica-Bold",
            textColor=C_WHITE, leading=12)),
    ]]
    ep_tbl = Table(ep_data, colWidths=[50*mm, 130*mm])
    ep_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), epc),
        ("TOPPADDING",   (0,0),(-1,-1), 3*mm),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3*mm),
        ("LEFTPADDING",  (0,0),(-1,-1), 6*mm),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    elems.append(ep_tbl)
    return elems


# ── Section header ────────────────────────────────────────
def _section_header(styles, title):
    tbl = Table([[Paragraph(title, styles["section_hdr"])]],
                colWidths=[A4[0] - 30*mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",    (0,0),(-1,-1), 4*mm),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4*mm),
        ("LEFTPADDING",   (0,0),(-1,-1), 6*mm),
    ]))
    return tbl


# ── Defect card ───────────────────────────────────────────
def _defect_card(styles, card, idx):
    sev   = card.get("severity", "Minor")
    sc    = severity_color(sev)
    conf  = int((card.get("confidence", 0)) * 100)
    W     = A4[0] - 30*mm
    col2  = (W - 2*mm) / 2

    def lv(label, value, mono=False):
        vs = styles["field_mono"] if mono else styles["field_value"]
        return [Paragraph(label, styles["field_label"]),
                Paragraph(str(value), vs)]

    # Header row
    hdr_data = [[
        Paragraph(card.get("defect_id", f"DFT-{idx+1:03d}"), ParagraphStyle(
            "dh1", fontSize=8, fontName="Helvetica", textColor=C_WHITE, leading=11)),
        Paragraph(card.get("defect_type", "—"), ParagraphStyle(
            "dh2", fontSize=10, fontName="Helvetica-Bold", textColor=C_WHITE, leading=13)),
        Paragraph(sev.upper(), ParagraphStyle(
            "dh3", fontSize=8, fontName="Helvetica-Bold", textColor=C_WHITE,
            leading=11, alignment=TA_RIGHT)),
    ]]
    hdr = Table(hdr_data, colWidths=[30*mm, W-75*mm, 40*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), sc),
        ("TOPPADDING",   (0,0),(-1,-1), 3*mm),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3*mm),
        ("LEFTPADDING",  (0,0),(-1,-1), 5*mm),
        ("RIGHTPADDING", (2,0),(2,0),  5*mm),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))

    # Grid fields
    ipc_c = card.get("ipc_classification","—")
    ipc_r = card.get("ipc_reference","—")

    fields_data = [
        [*lv("LOCATION (BBOX)", card.get("location","—"), mono=True),
         *lv("CONFIDENCE", f"{conf}%")],
        [*lv("IPC-A-610G CLASSIFICATION", ipc_c),
         *lv("IPC REFERENCE", ipc_r, mono=True)],
        [Paragraph("ROOT CAUSE", styles["field_label"]),
         Paragraph(card.get("root_cause","—"), styles["field_value"]),
         "", ""],
    ]
    fields_tbl = Table(fields_data, colWidths=[28*mm, col2-30*mm, 28*mm, col2-30*mm])
    fields_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_LGRAY),
        ("GRID",         (0,0),(-1,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
        ("SPAN",         (0,2),(3,2)),   # root cause spans full width
    ]))

    # Remediation
    rem_steps = card.get("remediation", [])
    rem_data  = [[
        Paragraph("RECOMMENDED REMEDIATION", styles["field_label"]),
    ]] + [
        [Paragraph(f"{i+1}.  {step}", styles["rem_item"])]
        for i, step in enumerate(rem_steps)
    ]
    rem_tbl = Table(rem_data, colWidths=[W])
    rem_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), C_WHITE),
        ("GRID",         (0,0),(-1,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("LEFTPADDING",  (0,0),(-1,-1), 5),
    ]))

    return [KeepTogether([hdr, fields_tbl, rem_tbl])]


# ── Evaluation section ────────────────────────────────────
def _evaluation_section(styles, ev, overall_pass):
    W    = A4[0] - 30*mm
    col3 = W / 3
    elems = []
    elems.append(_section_header(styles, "REPORT QUALITY EVALUATION"))
    elems.append(Spacer(1, 4*mm))

    # Overall strip
    epc  = C_GREEN if overall_pass else C_RED
    ovr_data = [[Paragraph(
        "✓  ALL EVALUATIONS PASSED — Report meets quality standards." if overall_pass else
        "✗  REVIEW REQUIRED — One or more evaluations failed. See details below.",
        ParagraphStyle("ov", fontSize=9, fontName="Helvetica-Bold",
                       textColor=C_WHITE, leading=13))]]
    ovr_tbl = Table(ovr_data, colWidths=[W])
    ovr_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), epc),
        ("TOPPADDING",   (0,0),(-1,-1), 4*mm),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4*mm),
        ("LEFTPADDING",  (0,0),(-1,-1), 6*mm),
    ]))
    elems += [ovr_tbl, Spacer(1, 6*mm)]

    # ── RAGAs ──
    ragas   = ev.get("ragas", {})
    ret     = ragas.get("retrieval",  {})
    gen     = ragas.get("generation", {})

    def pct(v): return f"{int((v or 0)*100)}%"

    ragas_data = [
        [Paragraph("RAGAs EVALUATION", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"])],
        # retrieval header
        [Paragraph("RETRIEVAL PIPELINE", styles["field_label"]),
         Paragraph("", styles["field_label"]),
         Paragraph("GENERATION PIPELINE", styles["field_label"]),
         Paragraph("", styles["field_label"])],
        [Paragraph("Context Precision", styles["field_value"]),
         Paragraph(pct(ret.get("context_precision")), styles["field_mono"]),
         Paragraph("Faithfulness", styles["field_value"]),
         Paragraph(pct(gen.get("faithfulness")), styles["field_mono"])],
        [Paragraph("Context Recall", styles["field_value"]),
         Paragraph(pct(ret.get("context_recall")), styles["field_mono"]),
         Paragraph("Answer Relevancy", styles["field_value"]),
         Paragraph(pct(gen.get("answer_relevancy")), styles["field_mono"])],
        [Paragraph("Average", styles["body_bold"]),
         Paragraph(pct(ret.get("average")), styles["field_mono"]),
         Paragraph("Average", styles["body_bold"]),
         Paragraph(pct(gen.get("average")), styles["field_mono"])],
        [Paragraph(f"Result: {ret.get('label','—')}", ParagraphStyle(
             "rr", fontSize=9, fontName="Helvetica-Bold",
             textColor=eval_color(ret.get("pass",False)))),
         Paragraph("", styles["field_label"]),
         Paragraph(f"Result: {gen.get('label','—')}", ParagraphStyle(
             "gr", fontSize=9, fontName="Helvetica-Bold",
             textColor=eval_color(gen.get("pass",False)))),
         Paragraph("", styles["field_label"])],
    ]
    ragas_tbl = Table(ragas_data, colWidths=[col3*0.6, col3*0.4, col3*0.6, col3*0.4])
    ragas_tbl.setStyle(TableStyle([
        ("SPAN",         (0,0),(3,0)),
        ("BACKGROUND",   (0,0),(3,0), C_DARK),
        ("BACKGROUND",   (0,1),(3,-1), C_LGRAY),
        ("GRID",         (0,1),(3,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(3,0), 6*mm),
    ]))
    elems += [ragas_tbl, Spacer(1, 2*mm)]

    # Interpretation
    ri = ret.get("interpretation","")
    gi = gen.get("generation","") or gen.get("interpretation","")
    if ri: elems.append(Paragraph(f"<b>Retrieval:</b> {ri}", styles["just_text"]))
    if gi: elems.append(Paragraph(f"<b>Generation:</b> {gi}", styles["just_text"]))
    elems.append(Spacer(1, 6*mm))

    # ── LLM Judge ──
    judge   = ev.get("llm_judge", {})
    overall = judge.get("overall", {})
    dims    = [
        ("ipc_grounding",       "IPC Grounding"),
        ("defect_coverage",     "Defect Coverage"),
        ("root_cause_quality",  "Root Cause Quality"),
        ("remediation_quality", "Remediation Quality"),
        ("hallucination_risk",  "Hallucination Risk"),
    ]
    judge_rows = [
        [Paragraph("LLM JUDGE EVALUATION", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"])],
        [Paragraph("DIMENSION", styles["field_label"]),
         Paragraph("SCORE", styles["field_label"]),
         Paragraph("RESULT", styles["field_label"]),
         Paragraph("JUSTIFICATION", styles["field_label"])],
    ]
    for key, label in dims:
        d = judge.get(key, {})
        s = d.get("score", 0)
        p = d.get("pass", False)
        j = d.get("justification", "—")
        judge_rows.append([
            Paragraph(label, styles["field_value"]),
            Paragraph(f"{s}/5", ParagraphStyle("js", fontSize=9, fontName="Helvetica-Bold",
                      textColor=eval_color(p), leading=12)),
            Paragraph(d.get("label","—"), ParagraphStyle("jp", fontSize=8, fontName="Helvetica-Bold",
                      textColor=eval_color(p), leading=11)),
            Paragraph(j, styles["just_text"]),
        ])
    # Overall row
    os  = overall.get("score", 0)
    op  = overall.get("pass", False)
    judge_rows.append([
        Paragraph("OVERALL", styles["body_bold"]),
        Paragraph(f"{os}/5", ParagraphStyle("jos", fontSize=10, fontName="Helvetica-Bold",
                  textColor=eval_color(op), leading=13)),
        Paragraph(overall.get("label","—"), ParagraphStyle("jop", fontSize=9, fontName="Helvetica-Bold",
                  textColor=eval_color(op), leading=12)),
        Paragraph(overall.get("summary","—"), styles["just_text"]),
    ])

    judge_tbl = Table(judge_rows, colWidths=[45*mm, 18*mm, 18*mm, W-81*mm])
    judge_tbl.setStyle(TableStyle([
        ("SPAN",         (0,0),(3,0)),
        ("BACKGROUND",   (0,0),(3,0),  C_DARK),
        ("BACKGROUND",   (0,1),(3,1),  C_BGRAY),
        ("BACKGROUND",   (0,2),(3,-2), C_LGRAY),
        ("BACKGROUND",   (0,-1),(3,-1),C_BGRAY),
        ("GRID",         (0,1),(3,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(3,0),  6*mm),
        ("LEFTPADDING",  (0,1),(-1,-1), 5),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    elems += [judge_tbl, Spacer(1, 6*mm)]

    # ── Citations ──
    cit      = ev.get("citations", {})
    cit_rows_data = cit.get("results", [])
    cit_rows = [
        [Paragraph("CITATION VERIFICATION", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"]),
         Paragraph("", styles["section_hdr"])],
        [Paragraph("DEFECT ID", styles["field_label"]),
         Paragraph("CITED REFERENCE", styles["field_label"]),
         Paragraph("VERDICT", styles["field_label"]),
         Paragraph("ACCEPTABILITY LANGUAGE", styles["field_label"]),
         Paragraph("REASON", styles["field_label"])],
    ]
    for r in cit_rows_data:
        vd    = r.get("verdict","FAIL")
        vc2   = C_GREEN if vd=="PASS" else C_AMBER if vd=="PARTIAL" else C_RED
        kws   = ", ".join(r.get("found_keywords",[])[:3]) or "none found"
        cit_rows.append([
            Paragraph(r.get("defect_id","?"),   styles["field_mono"]),
            Paragraph(r.get("cited_reference","—"), styles["field_mono"]),
            Paragraph(vd, ParagraphStyle("cv", fontSize=8, fontName="Helvetica-Bold",
                      textColor=vc2, leading=11)),
            Paragraph(kws, styles["just_text"]),
            Paragraph(r.get("reason","—")[:120], styles["just_text"]),
        ])

    # Score summary row
    score = cit.get("score", 1.0)
    cpas  = cit.get("pass", True)
    cit_rows.append([
        Paragraph("OVERALL", styles["body_bold"]),
        Paragraph(f"{int(score*100)}%", ParagraphStyle("cs2", fontSize=10, fontName="Helvetica-Bold",
                  textColor=eval_color(cpas), leading=13)),
        Paragraph(cit.get("label","—"), ParagraphStyle("cl", fontSize=9, fontName="Helvetica-Bold",
                  textColor=eval_color(cpas), leading=12)),
        Paragraph(cit.get("summary","—"), styles["just_text"]),
        Paragraph("", styles["just_text"]),
    ])

    cit_tbl = Table(cit_rows, colWidths=[22*mm, 42*mm, 18*mm, 40*mm, W-122*mm])
    cit_tbl.setStyle(TableStyle([
        ("SPAN",         (0,0),(4,0)),
        ("BACKGROUND",   (0,0),(4,0),  C_DARK),
        ("BACKGROUND",   (0,1),(4,1),  C_BGRAY),
        ("BACKGROUND",   (0,2),(4,-2), C_LGRAY),
        ("BACKGROUND",   (0,-1),(4,-1),C_BGRAY),
        ("GRID",         (0,1),(4,-1), 0.5, C_BGRAY),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(4,0),  6*mm),
        ("LEFTPADDING",  (0,1),(-1,-1), 5),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    elems += [cit_tbl]
    return elems


# ── Route ─────────────────────────────────────────────────
@pdf_bp.route("/download-report", methods=["POST"])
def download_report():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Send the /analyze JSON response as the request body."}), 400

    try:
        pdf_bytes = build_pdf(data)
        rid       = data.get("report_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        filename  = f"VLDA_Report_{rid}.pdf"
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype        = "application/pdf",
            as_attachment   = True,
            download_name   = filename,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
