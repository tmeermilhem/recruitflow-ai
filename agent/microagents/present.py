"""Generate run outputs: final.json (machine) and report.md (human) for top candidates."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import indent
from typing import Any, Dict, List

from agent.schemas import CandidateProfile, JobListing, QAItem, ScoreBreakdown

# Create a timestamped output directory under data/runs
def _run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("data") / "runs" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d

# Build a Markdown summary for one candidate (scores, skills, experience, education, Q&A)
def _summarize_candidate_md(i: int, cand: CandidateProfile, qa: List[QAItem], sc: ScoreBreakdown) -> str:
    lines: List[str] = []
    lines.append(f"## {i}. {cand.name or cand.doc_id}")
    lines.append(f"**Doc ID:** `{cand.doc_id}`  \n**Vector source:** `{cand.source_path or ''}`")
    if cand.contact:
        c = cand.contact
        parts = []
        if c.email: parts.append(f"📧 {c.email}")
        if c.phone: parts.append(f"📞 {c.phone}")
        if c.location: parts.append(f"📍 {c.location}")
        if c.linkedin: parts.append(f"🔗 LinkedIn: {c.linkedin}")
        if c.github: parts.append(f"💻 GitHub: {c.github}")
        if parts:
            lines.append("\n".join(parts))

    # Scores
    lines.append("\n**Scores**")
    lines.append(f"- Skills: {sc.skills_score} — {sc.reasons.get('skills','')}")
    lines.append(f"- Experience: {sc.experience_score} — {sc.reasons.get('experience','')}")
    lines.append(f"- Education: {sc.education_score} — {sc.reasons.get('education','')}")
    lines.append(f"- **Overall: {sc.overall_score}** — {sc.reasons.get('overall','')}")

    # Skills
    if cand.skills:
        lines.append("\n**Skills**")
        lines.append(", ".join(sorted(set(cand.skills))))

    # Experience
    if cand.experience:
        lines.append("\n**Experience**")
        for e in cand.experience:
            title = e.title or ""
            comp = e.company or ""
            dates = " ".join(filter(None, [e.start, "-", e.end]))
            lines.append(f"- **{title}** @ {comp} ({dates})")
            for b in e.bullets or []:
                lines.append(f"  - {b}")

    # Education
    if cand.education:
        lines.append("\n**Education**")
        for ed in cand.education:
            lines.append(f"- {ed.degree or ''} in {ed.field or ''} — {ed.school or ''} ({ed.year or ''})")

    # Q&A
    if qa:
        lines.append("\n**Questionnaire & Answers**")
        for j, q in enumerate(qa, 1):
            lines.append(f"{j}. **Q:** {q.question}")
            lines.append(f"   **A:** {q.answer}")

    # Raw
    if cand.raw_text:
        lines.append("\n<details>\n<summary>Raw Resume Text</summary>\n\n")
        lines.append(indent(cand.raw_text.strip(), "    "))
        lines.append("\n</details>")

    return "\n".join(lines).strip() + "\n"

def save_outputs(
    job: JobListing,
    top_items: List[Dict[str, Any]],
    out_dir: Path | None = None
) -> Dict[str, str]:
    """
    Writes:
      - final.json  (machine-friendly)
      - report.md   (human-friendly)
    Returns paths as strings.
    """
    out_dir = out_dir or _run_dir()

    # final.json
    final_blob = {
        "job": job.model_dump(),
        "top": [
            {
                "doc_id": it["doc_id"],
                "scores": it["scores"].model_dump(),
                "candidate": it["candidate"].model_dump(),
                "qa": [q.model_dump() for q in it["qa"]],
            }
            for it in top_items
        ],
    }
    (out_dir / "final.json").write_text(json.dumps(final_blob, indent=2, ensure_ascii=False), encoding="utf-8")

    # report.md
    md: List[str] = []
    md.append(f"# RecruitFlow AI — Top {len(top_items)} Candidates\n")
    md.append("## Job Listing (Parsed)\n")
    md.append(f"- **Title:** {job.title}")
    if job.required_skills: md.append(f"- **Required skills:** {', '.join(job.required_skills)}")
    if job.experience_required: md.append(f"- **Experience:** {job.experience_required}")
    if job.education_required: md.append(f"- **Education:** {job.education_required}")
    if job.extra_attributes: md.append(f"- **Extras:** {', '.join(job.extra_attributes)}")
    md.append("\n---\n")

    for i, it in enumerate(top_items, 1):
        md.append(_summarize_candidate_md(i, it["candidate"], it["qa"], it["scores"]))
        md.append("\n---\n")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    return {
        "final_json": str(out_dir / "final.json"),
        "report_md": str(out_dir / "report.md"),
        "run_dir": str(out_dir),
    }