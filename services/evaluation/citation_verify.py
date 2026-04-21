"""
Evaluation 4 — IPC Citation Verification

Pass rule (both conditions must hold):
  1. SEMANTIC MATCH — the cited IPC clause must be within cosine distance
     threshold of the defect claim (i.e. it is actually about that defect).
  2. ACCEPTABILITY LANGUAGE — the matched chunk must contain IPC acceptability
     criteria language (Target / Acceptable / Defect conditions, class levels,
     shall/must-not, or inspection directives).

Rationale: IPC-based inspection is about matching observed conditions to
acceptability criteria — not just finding any paragraph that mentions the
defect type. A chunk discussing the cause of solder bridges without stating
a class-level verdict does NOT support a defect claim under IPC-A-610G.

Scoring:
  Each defect card gets: PASS / PARTIAL / FAIL
    PASS    — both conditions met
    PARTIAL — semantically matched but no acceptability language found
    FAIL    — no semantic match at all (possible hallucination)

  Overall score = (PASS*1 + PARTIAL*0.5) / total
  Overall PASS  = score >= 0.80
"""

from services.rag_service import get_collection
from models.embed_model   import encode
from config import CITATION_DISTANCE_THRESHOLD, IPC_ACCEPTABILITY_KEYWORDS

OVERALL_PASS_THRESHOLD = 0.80


def run(defect_cards: list[dict]) -> dict:
    if not defect_cards:
        return _empty()

    collection = get_collection()
    results    = []

    for card in defect_cards:
        ref         = card.get("ipc_reference", "").strip()
        defect_type = card.get("defect_type", "")

        if not ref or ref == "—":
            results.append(_entry(card, ref, "FAIL",
                reason="No IPC reference cited in the report entry.",
                acceptability_found=False))
            continue

        try:
            # Query using both the cited reference AND the defect type
            # to catch cases where Claude paraphrases a section title
            query = f"{ref} {defect_type} acceptability"
            hits  = collection.query(
                query_embeddings = encode([query]),
                n_results        = 1,
                include          = ["documents", "metadatas", "distances"],
            )
            distance   = hits["distances"][0][0]
            chunk_text = hits["documents"][0][0]
            page       = hits["metadatas"][0][0].get("page", "?")

            # Condition 1: semantic match
            semantically_matched = distance < CITATION_DISTANCE_THRESHOLD

            # Condition 2: acceptability language present in the matched chunk
            chunk_lower          = chunk_text.lower()
            found_keywords       = [kw for kw in IPC_ACCEPTABILITY_KEYWORDS if kw in chunk_lower]
            acceptability_found  = len(found_keywords) > 0

            if semantically_matched and acceptability_found:
                verdict = "PASS"
                reason  = (f"Cited clause matched (distance {round(distance,3)}) and contains "
                           f"IPC acceptability language: {', '.join(found_keywords[:3])}.")
            elif semantically_matched and not acceptability_found:
                verdict = "PARTIAL"
                reason  = (f"Cited clause is semantically related (distance {round(distance,3)}) "
                           f"but the matched chunk contains no acceptability criteria "
                           f"(no Target/Acceptable/Defect condition language found). "
                           f"The citation may reference a descriptive section rather than a "
                           f"workmanship criterion — verify manually against IPC-A-610G.")
            else:
                verdict = "FAIL"
                reason  = (f"No matching IPC clause found (distance {round(distance,3)} exceeds "
                           f"threshold {CITATION_DISTANCE_THRESHOLD}). Reference '{ref}' may be "
                           f"fabricated or significantly paraphrased from the standard.")

            results.append(_entry(card, ref, verdict, reason,
                                  acceptability_found, page,
                                  chunk_text[:250] if semantically_matched else None,
                                  found_keywords))

        except Exception as e:
            results.append(_entry(card, ref, "FAIL", f"Query error: {e}",
                                  acceptability_found=False))

    # Overall score
    score = sum(1.0 if r["verdict"] == "PASS" else 0.5 if r["verdict"] == "PARTIAL" else 0.0
                for r in results) / len(results)
    score = round(score, 3)
    passed = score >= OVERALL_PASS_THRESHOLD

    return {
        "results":  results,
        "score":    score,
        "pass":     passed,
        "label":    "PASS" if passed else "FAIL",
        "summary":  _summary(score, results, passed),
        "error":    None,
    }


# ── Helpers ───────────────────────────────────────────────
def _entry(card, ref, verdict, reason,
           acceptability_found=False,
           matched_page=None,
           matched_excerpt=None,
           found_keywords=None) -> dict:
    return {
        "defect_id":           card.get("defect_id", "?"),
        "defect_type":         card.get("defect_type", "?"),
        "cited_reference":     ref,
        "verdict":             verdict,      # PASS / PARTIAL / FAIL
        "reason":              reason,
        "acceptability_found": acceptability_found,
        "found_keywords":      found_keywords or [],
        "matched_page":        matched_page,
        "matched_excerpt":     matched_excerpt,
    }


def _summary(score: float, results: list, passed: bool) -> str:
    n_pass    = sum(1 for r in results if r["verdict"] == "PASS")
    n_partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
    n_fail    = sum(1 for r in results if r["verdict"] == "FAIL")
    total     = len(results)

    lines = [f"{n_pass}/{total} citations fully verified, "
             f"{n_partial} partial (no acceptability language), "
             f"{n_fail} failed (possible hallucination)."]

    if n_partial:
        lines.append("PARTIAL citations reference IPC sections that discuss the defect "
                     "but do not state a class-level workmanship verdict — they cannot "
                     "alone justify an Accept/Reject decision.")
    if n_fail:
        lines.append("FAIL citations could not be matched to the knowledge base. "
                     "Do not rely on these clauses without manual IPC-A-610G verification.")
    if passed:
        lines.append("Overall citation quality is sufficient for a preliminary inspection report.")
    else:
        lines.append("Overall citation quality is below threshold. "
                     "Human review against IPC-A-610G is required before acting on this report.")

    return " ".join(lines)


def _empty() -> dict:
    return {"results": [], "score": 1.0, "pass": True,
            "label": "PASS", "summary": "No defect cards to verify.", "error": None}
