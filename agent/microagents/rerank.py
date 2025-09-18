# agent/microagents/rerank.py
from __future__ import annotations
from typing import List, Dict, Any
from dataclasses import asdict

from agent.schemas import CandidateProfile, QAItem, ScoreBreakdown

def compile_results(
    candidates: List[Dict[str, Any]],
    qa_pairs: Dict[str, List[QAItem]],
    scores: Dict[str, ScoreBreakdown],
) -> List[Dict[str, Any]]:
    """
    candidates: list of {"id": str, "score": float, "candidate": CandidateProfile}
    qa_pairs:   { doc_id: [QAItem, ...] }
    scores:     { doc_id: ScoreBreakdown }

    Returns a flat list where each item is easy to sort and present.
    """
    out: List[Dict[str, Any]] = []
    for row in candidates:
        cand: CandidateProfile = row["candidate"]
        doc_id = cand.doc_id
        out.append({
            "doc_id": doc_id,
            "vector_score": float(row.get("score", 0.0)),
            "candidate": cand,
            "qa": qa_pairs.get(doc_id, []),
            "scores": scores.get(doc_id, None),
        })
    return out

def rerank_top(results: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Sort primarily by overall_score (desc), then by vector_score (desc) as tiebreaker.
    """
    def sort_key(r: Dict[str, Any]):
        sb: ScoreBreakdown | None = r["scores"]
        overall = sb.overall_score if sb else -1
        vec = r.get("vector_score", 0.0)
        return (overall, vec)

    ranked = sorted(results, key=sort_key, reverse=True)
    return ranked[:top_n]