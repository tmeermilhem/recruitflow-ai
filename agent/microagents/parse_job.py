"""Normalize a raw job listing to structured JSON via LLM, with a regex-based fallback."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from agent.config import SETTINGS
from agent.llm import chat_json_with_logging
from agent.schemas import JobListing

# System prompt: define expected JSON schema and enforce JSON-only output.
SYSTEM = (
    "You convert a raw job listing into a strict JSON object with fields:\n"
    '{\n'
    '  "title": str,\n'
    '  "required_skills": [str],\n'
    '  "experience_required": str|null,\n'
    '  "education_required": str|null,\n'
    '  "extra_attributes": [str]\n'
    '}\n'
    "Keep it concise. No commentary. Return JSON only."
)

# User template: inject raw job text and specify defaults for missing fields.
USER_TMPL = (
    "Raw job listing:\n---\n{job_text}\n---\n"
    "Extract the fields. If missing, use null or empty lists as appropriate."
)

# Heuristic fallback used when LLM output is missing/invalid JSON.
def _fallback_parse(job_text: str) -> Dict[str, Any]:
    title = None
    first_line = job_text.strip().splitlines()[0].strip() if job_text.strip().splitlines() else ""
    if 5 <= len(first_line) <= 120:
        title = first_line
    skills = []
    skill_match = re.search(r"skills?\s*[:\-]\s*(.+)", job_text, flags=re.I)
    if skill_match:
        raw = skill_match.group(1)
        skills = [s.strip().lower() for s in re.split(r"[,/|•;]", raw) if len(s.strip()) >= 2]
        skills = list(dict.fromkeys(skills))
    return {
        "title": title or "Job",
        "required_skills": skills,
        "experience_required": None,
        "education_required": None,
        "extra_attributes": []
    }

# Main entry: parse raw job text into a JobListing via LLM, with fallback.
def parse_job_to_json(job_text: str) -> JobListing:
    """
    Uses team12-gpt4o to convert raw text into JobListing schema.
    Token usage is logged (prompt vs completion). Falls back to heuristics if invalid JSON.
    """
    user = USER_TMPL.format(job_text=job_text)
    content, _resp = chat_json_with_logging(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        model=SETTINGS.CHAT_DEPLOYMENT,
        tag="parse_job",
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    try:
        data = json.loads(content)
    except Exception:
        data = _fallback_parse(job_text)
    return JobListing(**data)