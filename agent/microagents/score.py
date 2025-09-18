"""Evaluate a candidate against a job via LLM and return a ScoreBreakdown."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.config import SETTINGS
from agent.llm import chat_json_with_logging
from agent.schemas import CandidateProfile, JobListing, QAItem, ScoreBreakdown

# System prompt: enforce JSON scoring schema with 0–100 ranges
SYSTEM = (
    "You are a strict evaluator. Score a candidate (0-100) against a job listing.\n"
    "Return ONLY JSON with fields:\n"
    "{\n"
    '  "skills_score": int,     // 0..100\n'
    '  "experience_score": int, // 0..100\n'
    '  "education_score": int,  // 0..100\n'
    '  "overall_score": int,    // 0..100\n'
    '  "reasons": {\n'
    '    "skills": "short reason",\n'
    '    "experience": "short reason",\n'
    '    "education": "short reason",\n'
    '    "overall": "short reason"\n'
    "  }\n"
    "}\n"
    "Be concise and evidence-based. If information is missing, penalize accordingly."
)

# User template: inject job, candidate, and Q&A as JSON
USER_TMPL = (
    "JOB (JSON):\n{job_json}\n\n"
    "CANDIDATE RESUME (JSON):\n{cand_json}\n\n"
    "Q&A (JSON array of {{question, answer}}):\n{qa_json}\n\n"
    "Score the candidate against the job. Follow the output schema exactly."
)

# Helper: coerce value to int within [0, 100]
def _coerce_int(x, floor=0, ceil=100) -> int:
    try:
        v = int(round(float(x)))
    except Exception:
        v = 0
    return max(floor, min(ceil, v))

# Call LLM, parse JSON, and build ScoreBreakdown
def score_candidate(job: JobListing, cand: CandidateProfile, qa_pairs: List[QAItem]) -> ScoreBreakdown:
    job_json = json.dumps(job.model_dump(), ensure_ascii=False)
    cand_json = json.dumps(cand.model_dump(), ensure_ascii=False)
    qa_json = json.dumps([q.model_dump() for q in qa_pairs], ensure_ascii=False)

    user = USER_TMPL.format(job_json=job_json, cand_json=cand_json, qa_json=qa_json)

    content, _resp = chat_json_with_logging(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        model=SETTINGS.CHAT_DEPLOYMENT,  # team12-gpt4o
        tag="scoring",
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    # Parse + coerce into ScoreBreakdown
    try:
        data = json.loads(content)
    except Exception:
        data = {}

    skills_score = _coerce_int(data.get("skills_score", 0))
    experience_score = _coerce_int(data.get("experience_score", 0))
    education_score = _coerce_int(data.get("education_score", 0))
    overall_score = _coerce_int(data.get("overall_score", 0))

    # reasons block (short strings)
    reasons_in = data.get("reasons", {}) if isinstance(data.get("reasons", {}), dict) else {}
    reasons = {
        "skills": str(reasons_in.get("skills", ""))[:240],
        "experience": str(reasons_in.get("experience", ""))[:240],
        "education": str(reasons_in.get("education", ""))[:240],
        "overall": str(reasons_in.get("overall", ""))[:240],
    }

    return ScoreBreakdown(
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        overall_score=overall_score,
        reasons=reasons,
    )