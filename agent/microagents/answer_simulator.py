# agent/microagents/answer_simulator.py
from __future__ import annotations
from typing import List
import json

from agent.clients import get_llama_client
from agent.config import SETTINGS
from agent.schemas import JobListing, CandidateProfile, QAItem
from agent.utils_llama import append_usage_llama

# System instruction guiding the model to return ONLY JSON with an 'answers' array.
SYSTEM = (
    "You answer candidate questionnaires realistically. "
    "Return ONLY JSON with field 'answers': [string,...], aligned one-to-one to the provided questions. "
    "Be concise, honest, and grounded in the candidate resume. If a fact is unknown, say so briefly."
)

# User message template injecting job, candidate, and questions; requests strict JSON.
USER_TMPL = (
    "JOB (JSON):\n{job_json}\n\n"
    "CANDIDATE (JSON):\n{cand_json}\n\n"
    "QUESTIONS (JSON array):\n{questions_json}\n\n"
    'Respond with JSON: {{"answers": [ ... ]}} matching the question order.'
)

def simulate_answers(job: JobListing, cand: CandidateProfile, questions: List[QAItem]) -> List[QAItem]:
    """Call the LLM to generate concise answers to the given questions.

Args:
    job: JobListing containing role requirements and context.
    cand: CandidateProfile with resume-derived data.
    questions: Ordered list of QAItem questions to answer.

Returns:
    List[QAItem]: same order as input, with the 'answer' field populated.

Notes:
    - Enforces JSON response via response_format to reduce parsing errors.
    - Pads/truncates answers to match question count; falls back to synthetic
      placeholders if parsing fails.
    - Logs token usage with append_usage_llama for cost tracking.
"""
    
    llama = get_llama_client()

    job_json = json.dumps(job.model_dump(), ensure_ascii=False)
    cand_json = json.dumps(cand.model_dump(), ensure_ascii=False)
    qs = [q.question for q in questions]
    questions_json = json.dumps(qs, ensure_ascii=False)

    user = USER_TMPL.format(job_json=job_json, cand_json=cand_json, questions_json=questions_json)

    resp = llama.chat.completions.create(
        model=SETTINGS.LLAMA_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.6,
    )

    # --- log tokens to the LLaMA-only counters ---
    usage = getattr(resp, "usage", None) or {}
    prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
    completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
    total_tokens = getattr(usage, "total_tokens", None)
    append_usage_llama(
        kind="chat",
        model=SETTINGS.LLAMA_DEPLOYMENT,
        tag="answer_simulation",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    # ---------------------------------------------

    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        answers = data.get("answers", [])
        # Align defensively: pad/truncate to match question length
        if len(answers) < len(questions):
            answers = answers + ["(no answer)"] * (len(questions) - len(answers))
        elif len(answers) > len(questions):
            answers = answers[:len(questions)]
    except Exception:
        # very simple fallback if JSON fails
        answers = [f"(synthetic) Answer to: {q.question[:80]}..." for q in questions]

    # Return QAItems with answers filled
    out: List[QAItem] = []
    for q, a in zip(questions, answers):
        out.append(QAItem(question=q.question, answer=(a or "").strip()))
    return out