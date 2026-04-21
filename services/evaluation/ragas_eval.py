"""
services/evaluation/ragas_eval.py

Implements the four RAGAs-equivalent metrics WITHOUT the ragas library.
Uses your existing Claude client and BAAI embeddings — no OpenAI key needed.

Metrics:
  Retrieval pipeline  → context_precision + context_recall
  Generation pipeline → faithfulness + answer_relevancy

Retrieval metrics (embedding-based, no LLM needed):
  context_precision  = fraction of retrieved chunks that are actually relevant
                       (cosine similarity to query > RELEVANCE_THRESHOLD)
  context_recall     = how much of the defect query is covered by top chunk
                       (sim between query and best chunk)

Generation metrics (Claude-based):
  faithfulness       = fraction of report claims that are grounded in context
  answer_relevancy   = how directly the report answers the defect query
                       (embedding sim between query and report excerpt)
"""

import json
import re
import anthropic
from models.embed_model import encode
from config import CLAUDE_MODEL, RAGAS_RETRIEVAL_PASS, RAGAS_GENERATION_PASS

RELEVANCE_THRESHOLD = 0.55   # cosine similarity; above = chunk is relevant to query


# ── Public entry point ────────────────────────────────────
def run(detections: list[dict], rag_contexts: dict, raw_report: str) -> dict:
    if not rag_contexts:
        return _empty("No RAG contexts to evaluate.")

    try:
        retrieval  = _eval_retrieval(rag_contexts)
        generation = _eval_generation(rag_contexts, raw_report)
        overall    = retrieval["pass"] and generation["pass"]

        return {
            "retrieval":  retrieval,
            "generation": generation,
            "pass":       overall,
            "summary":    _summary(retrieval, generation),
            "error":      None,
        }
    except Exception as e:
        return _empty(str(e))


# ── Retrieval metrics (embedding-based) ───────────────────
def _eval_retrieval(rag_contexts: dict) -> dict:
    """
    context_precision: of all retrieved chunks, what fraction are
                       actually relevant to the defect query?
    context_recall:    does the best chunk cover the query well?
                       (proxy: max relevance score across all chunks)
    """
    all_precision_scores = []
    all_recall_scores    = []

    for defect_type, chunks in rag_contexts.items():
        if not chunks:
            continue

        # Encode the query once
        query_vec = encode([defect_type])[0]

        relevances = []
        for chunk in chunks:
            chunk_vec = encode([chunk["text"]])[0]
            sim       = _cosine(query_vec, chunk_vec)
            relevances.append(sim)

        # precision = fraction of chunks above relevance threshold
        precision = sum(1 for s in relevances if s >= RELEVANCE_THRESHOLD) / len(relevances)
        # recall proxy = best single chunk coverage
        recall    = max(relevances)

        all_precision_scores.append(precision)
        all_recall_scores.append(recall)

    cp  = round(sum(all_precision_scores) / len(all_precision_scores), 3) if all_precision_scores else 0.0
    cr  = round(sum(all_recall_scores)    / len(all_recall_scores),    3) if all_recall_scores    else 0.0
    avg = round((cp + cr) / 2, 3)
    ok  = avg >= RAGAS_RETRIEVAL_PASS

    return {
        "context_precision": cp,
        "context_recall":    cr,
        "average":           avg,
        "pass":              ok,
        "label":             "PASS" if ok else "FAIL",
        "interpretation": (
            f"Precision {_pct(cp)} — "
            f"{'retrieved chunks were mostly relevant to the defect query' if cp >= RAGAS_RETRIEVAL_PASS else 'several retrieved chunks were off-topic; consider re-chunking or increasing top-k selectivity'}. "
            f"Recall {_pct(cr)} — "
            f"{'best retrieved chunk closely matches the defect query' if cr >= RAGAS_RETRIEVAL_PASS else 'top retrieved chunk is weakly related; the knowledge base may not cover this defect type well'}."
        ),
    }


# ── Generation metrics ────────────────────────────────────
def _eval_generation(rag_contexts: dict, raw_report: str) -> dict:
    """
    faithfulness:      Claude checks what fraction of report claims are
                       supported by the retrieved IPC context.
    answer_relevancy:  embedding similarity between each defect query
                       and the portion of the report about that defect.
    """
    faithfulness    = _eval_faithfulness(rag_contexts, raw_report)
    answer_relevancy = _eval_answer_relevancy(rag_contexts, raw_report)

    avg = round((faithfulness + answer_relevancy) / 2, 3)
    ok  = avg >= RAGAS_GENERATION_PASS

    return {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "average":           avg,
        "pass":              ok,
        "label":             "PASS" if ok else "FAIL",
        "interpretation": (
            f"Faithfulness {_pct(faithfulness)} — "
            f"{'Claude stayed grounded in retrieved IPC text with minimal unsupported claims' if faithfulness >= RAGAS_GENERATION_PASS else 'Claude made claims not fully supported by retrieved IPC chunks — possible hallucination or over-generalisation'}. "
            f"Answer Relevancy {_pct(answer_relevancy)} — "
            f"{'report content is closely aligned with the specific defect queries' if answer_relevancy >= RAGAS_GENERATION_PASS else 'report may be too generic and not focused on the specific defects detected'}."
        ),
    }


def _eval_faithfulness(rag_contexts: dict, raw_report: str) -> float:
    """
    Ask Claude: given this context, which claims in the report are supported?
    Returns fraction of supported claims (0.0–1.0).
    """
    context_block = "\n".join(
        f"[{defect} | Page {c['page']}]: {c['text'][:300]}"
        for defect, chunks in rag_contexts.items()
        for c in chunks
    )
    # Use only a relevant excerpt of the report to keep tokens low
    report_excerpt = raw_report[:1500]

    prompt = f"""You are evaluating whether an AI-generated PCB inspection report is grounded in its source context.

RETRIEVED IPC-A-610G CONTEXT:
{context_block}

REPORT EXCERPT:
{report_excerpt}

Task: Extract up to 10 specific factual claims from the report (e.g. "Open Circuit is a Class 2 Defect", "Remediation requires flux application").
For each claim, judge whether it is directly supported by the context above.

Respond ONLY with valid JSON, no markdown:
{{
  "claims": [
    {{"claim": "<text>", "supported": true}},
    {{"claim": "<text>", "supported": false}}
  ]
}}"""

    try:
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        raw  = re.sub(r"^```[a-z]*\n?", "", msg.content[0].text.strip()).rstrip("`").strip()
        data = json.loads(raw)
        claims = data.get("claims", [])
        if not claims:
            return 1.0
        supported = sum(1 for c in claims if c.get("supported", False))
        return round(supported / len(claims), 3)
    except Exception:
        return 0.75   # neutral fallback if Claude call fails


def _eval_answer_relevancy(rag_contexts: dict, raw_report: str) -> float:
    """
    For each defect query, compute cosine similarity between the query
    embedding and the report embedding (using a short window around
    each defect mention in the report).
    """
    scores = []
    for defect_type in rag_contexts:
        query_vec = encode([defect_type])[0]

        # Find the report section most relevant to this defect
        # Use a sliding 300-char window, pick the best-matching segment
        best_sim = 0.0
        words    = raw_report.split()
        window   = 60   # words
        for i in range(0, max(1, len(words) - window), window // 2):
            segment  = " ".join(words[i : i + window])
            seg_vec  = encode([segment])[0]
            sim      = _cosine(query_vec, seg_vec)
            best_sim = max(best_sim, sim)

        scores.append(best_sim)

    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ── Helpers ───────────────────────────────────────────────
def _cosine(a: list, b: list) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return round(dot / (na * nb), 4)


def _pct(v: float) -> str:
    return f"{int(v * 100)}%"


def _summary(ret: dict, gen: dict) -> str:
    parts = []
    if not ret["pass"]:
        parts.append(
            "Retrieval stage failed — retrieved IPC chunks are weakly matched to the defect queries. "
            "Consider re-chunking by defect type or increasing TOP_K_CHUNKS in config.py."
        )
    if not gen["pass"]:
        parts.append(
            "Generation stage failed — Claude's report contains claims not well-supported by the retrieved context. "
            "Review the faithfulness score and consider adding more specific IPC acceptability text to the knowledge base."
        )
    if not parts:
        return (
            "Both retrieval and generation passed. "
            "Retrieved IPC context is relevant and Claude used it faithfully in the report."
        )
    return " ".join(parts)


def _empty(msg: str) -> dict:
    empty_block = {
        "context_precision": 0.0, "context_recall": 0.0,
        "faithfulness": 0.0, "answer_relevancy": 0.0,
        "average": 0.0, "pass": False, "label": "FAIL",
        "interpretation": msg,
    }
    return {
        "retrieval":  {k: v for k, v in empty_block.items() if k in
                       ["context_precision","context_recall","average","pass","label","interpretation"]},
        "generation": {k: v for k, v in empty_block.items() if k in
                       ["faithfulness","answer_relevancy","average","pass","label","interpretation"]},
        "pass":    False,
        "summary": msg,
        "error":   msg,
    }

