# Generate a candidate-specific questionnaire via LLM; returns List[QAItem].
from __future__ import annotations

import json
from typing import Any, Dict, List

from agent.config import SETTINGS
from agent.llm import chat_json_with_logging
from agent.schemas import CandidateProfile, JobListing, QAItem



# System prompt: JSON-only output with 'questions' array (target 6–8 items).
SYSTEM = (
    """You generate a concise, candidate-specific questionnaire as JSON.\n"
    "Return ONLY a JSON object with field 'questions': [string, ...].\n"
    "Cover: (1) experience clarification, (2) skill depth, (3) education/certs, "
    "(4) soft skills/scenarios, (5) 1–2 job-specific probes tied to the listing.\n"
    "Keep each question short and clear. 6–8 total."""
)

# User template: inject structured job and candidate JSON.
USER_TMPL = (
    "JOB LISTING (structured):\n"
    "{job_json}\n\n"
    "CANDIDATE (structured resume JSON):\n"
    "{cand_json}\n\n"
    "Produce questions targeted to this candidate and job."
)

# Public API: call LLM, enforce json_object, clamp count to n_min..n_max,
def generate_questions(job: JobListing, cand: CandidateProfile, n_min: int = 6, n_max: int = 8) -> List[QAItem]:
    user = USER_TMPL.format(
        job_json=json.dumps(job.model_dump(), ensure_ascii=False),
        cand_json=json.dumps(cand.model_dump(), ensure_ascii=False),
    )

    content, _resp = chat_json_with_logging(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        model=SETTINGS.CHAT_DEPLOYMENT,   # team12-gpt4o
        tag="questionnaire",
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    try:
        data = json.loads(content)
        qs = data.get("questions", [])
        # length control
        qs = [q for q in qs if isinstance(q, str) and q.strip()]
        if len(qs) < n_min:
            # pad with generic fallbacks if model returned too few
            fallbacks = [
                "Can you elaborate on a project that best aligns with this role?",
                "Which tools did you use most recently for data workflows?",
                "Describe a challenge you faced and how you resolved it.",
                "What impact did your work have on business KPIs?",
            ]
            for f in fallbacks:
                if len(qs) >= n_min: break
                qs.append(f)
        elif len(qs) > n_max:
            qs = qs[:n_max]
    except Exception:
        # extremely defensive fallback
        qs = [
            "Can you clarify your responsibilities in your most recent role?",
            "Which of your listed skills are strongest for this job?",
            "What relevant coursework or certifications do you have?",
            "Describe a time you solved a complex problem under time pressure.",
            "Have you implemented forecasting or optimization related to this role?",
            "How do you collaborate with cross-functional teams?"
        ]

    return [QAItem(question=q, answer="") for q in qs]