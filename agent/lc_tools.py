# agent/lc_tools.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from agent.microagents.score import score_candidate
from agent.microagents.answer_simulator import simulate_answers
from langchain.tools import tool
from typing import Optional
from agent.config import SETTINGS
from agent.schemas import JobListing, CandidateProfile, QAItem
from agent.microagents.parse_job import parse_job_to_json
from agent.microagents.retrieve import search_candidates
from agent.microagents.questionnaire import generate_questions
from agent.microagents.answer_simulator import simulate_answers as _simulate_answers
from agent.microagents.score import score_candidate as _score_candidate
from agent.microagents.rerank import rerank_top
from agent.microagents.present import save_outputs
from agent.schemas import JobListing, CandidateProfile, QAItem, ScoreBreakdown
# Session helpers
from agent.session import (
    # job
    set_job, get_job,
    # shortlist
    set_shortlist, get_shortlist,
    # current candidate & doc
    set_current_candidate, get_current_candidate,
    set_current_doc_id, get_current_doc_id,
    # QA (by doc + "current")
    set_qa_for, get_qa_for, get_qa_map,
    set_current_questions, get_current_questions,
    # scores (by doc)
    set_score_for, get_score_for, get_score_map,
    set_qa_map, set_score_map,
)

# ----------------------------
# 1) Parse Job
# ----------------------------
@tool("parse_job", return_direct=False)
def lc_parse_job(raw_job_text: str) -> dict:
    """Parse a raw job listing into structured JSON (title, skills, experience_required, education_required, extras)."""
    job = parse_job_to_json(raw_job_text)
    out = job.model_dump()
    set_job(out)  # remember for later tools
    return out


# ----------------------------
# 2) Retrieve Candidates
# ----------------------------
@tool("retrieve_candidates", return_direct=False)
def lc_retrieve_candidates(job_json: Optional[dict] = None, top_k: Optional[int] = None) -> list:
    """
    Retrieve top-K candidates from the vector DB. Returns:
      [{'id','score','candidate': <CandidateProfile as dict>}, ...]
    Falls back to last parsed job if job_json is omitted.
    """
    if job_json is None:
        job_json = get_job()
        if not job_json:
            raise ValueError("retrieve_candidates: no job_json provided and no parsed job in session. Run parse_job first.")

    if top_k is None:
        top_k = SETTINGS.TOP_K

    job = JobListing(**job_json)
    hits = search_candidates(job, top_k=top_k)

    out: List[Dict[str, Any]] = []
    for h in hits:
        cand: CandidateProfile = h["candidate"]
        out.append(
            {
                "id": h["id"],
                "score": h["score"],
                "candidate": cand.model_dump(),
            }
        )

    # store shortlist in session and set a default "current" candidate
    set_shortlist(out)
    if out:
        first_doc = out[0]["candidate"].get("doc_id")
        set_current_candidate(out[0]["candidate"])
        if first_doc:
            set_current_doc_id(first_doc)

    return out


# ----------------------------
# 3a) Generate Questionnaire
# ----------------------------
@tool("generate_questionnaire", return_direct=False)
def lc_generate_questionnaire(
    job_json: Optional[dict] = None,
    candidate_json: Optional[dict] = None,
    n_min: int = 6,
    n_max: int = 8,
) -> list:
    """
    Generate targeted questions for a single candidate.
    If job/candidate not provided, falls back to:
      - last parsed job
      - current candidate (or first in shortlist)
    Returns a list of {'question','answer': ''}.
    """
    # Resolve job
    if job_json is None:
        job_json = get_job()
        if not job_json:
            raise ValueError("generate_questionnaire: no job_json provided and no parsed job in session.")

    # Resolve candidate
    if candidate_json is None:
        candidate_json = get_current_candidate()
        if candidate_json is None:
            shortlist = get_shortlist() or []
            if not shortlist:
                raise ValueError("generate_questionnaire: no candidate_json provided and shortlist is empty. Run retrieve_candidates first.")
            candidate_json = shortlist[0]["candidate"]

    job = JobListing(**job_json)
    cand = CandidateProfile(**candidate_json)

    # Generate and cache
    qs = generate_questions(job, cand, n_min=n_min, n_max=n_max)
    qa = [q.model_dump() for q in qs]

    # remember as "current" and under candidate doc_id
    doc_id = cand.doc_id
    set_current_questions(qa)
    if doc_id:
        set_qa_for(doc_id, qa)

    return qa


# ----------------------------
# 3b) Simulate Answers
# ----------------------------
@tool("simulate_answers", return_direct=False)
def lc_simulate_answers(
    job_json: Optional[dict] = None,
    candidate_json: Optional[dict] = None,
    questions: Optional[list] = None,
) -> list:
    """
    Simulate realistic answers for a single candidate's questionnaire.
    Falls back to session for job, candidate, and questions if omitted.
    Returns the aligned list with 'answer' filled.
    """
    # Resolve job
    if job_json is None:
        job_json = get_job()
        if not job_json:
            raise ValueError("simulate_answers: no job_json provided and no parsed job in session.")

    # Resolve candidate
    if candidate_json is None:
        candidate_json = get_current_candidate()
        if candidate_json is None:
            shortlist = get_shortlist() or []
            if not shortlist:
                raise ValueError("simulate_answers: no candidate_json provided and shortlist is empty.")
            candidate_json = shortlist[0]["candidate"]

    cand = CandidateProfile(**candidate_json)
    doc_id = cand.doc_id

    # Resolve questions
    if questions is None or len(questions) == 0:
        # prefer per-doc questions, then current questions
        q_from_doc = get_qa_for(doc_id) if doc_id else []
        if q_from_doc:
            questions = q_from_doc
        else:
            questions = get_current_questions()
            if not questions:
                raise ValueError("simulate_answers: no questions provided and none found in session. Run generate_questionnaire first.")

    job = JobListing(**job_json)
    qa_items = [QAItem(**q) for q in questions]

    answered = _simulate_answers(job, cand, qa_items)
    qa = [a.model_dump() for a in answered]

    # persist back to session
    if doc_id:
        set_qa_for(doc_id, qa)
    set_current_questions(qa)

    return qa


# ----------------------------
# 3c) Score Candidate
# ----------------------------
@tool("score_candidate", return_direct=False)
def lc_score_candidate(
    job_json: Optional[dict] = None,
    candidate_json: Optional[dict] = None,
    qa_pairs: Optional[list] = None,
) -> dict:
    """
    Score a single candidate (skills/experience/education/overall + short reasons).
    Falls back to session for job, candidate, and QA pairs if omitted.
    """
    # Resolve job
    if job_json is None:
        job_json = get_job()
        if not job_json:
            raise ValueError("score_candidate: no job_json provided and no parsed job in session.")

    # Resolve candidate
    if candidate_json is None:
        candidate_json = get_current_candidate()
        if candidate_json is None:
            shortlist = get_shortlist() or []
            if not shortlist:
                raise ValueError("score_candidate: no candidate_json provided and shortlist is empty.")
            candidate_json = shortlist[0]["candidate"]

    cand = CandidateProfile(**candidate_json)
    doc_id = cand.doc_id

    # Resolve QA
    if qa_pairs is None or len(qa_pairs) == 0:
        qa_pairs = get_qa_for(doc_id) if doc_id else []
        if not qa_pairs:
            qa_pairs = get_current_questions()
        if not qa_pairs:
            raise ValueError("score_candidate: no qa_pairs provided and none found in session. Run simulate_answers first.")

    job = JobListing(**job_json)
    qa_items = [QAItem(**q) for q in qa_pairs]

    scored = _score_candidate(job, cand, qa_items)
    sc_dict = scored.model_dump()

    # persist score by doc
    if doc_id:
        set_score_for(doc_id, sc_dict)

    return sc_dict


# ----------------------------
# 4) Finalize and Save
# ----------------------------
@tool("finalize_and_save", return_direct=True)
def lc_finalize_and_save(
    job_json: dict | None = None,
    shortlisted: list | None = None,
    qa_by_doc: dict | None = None,
    scores_by_doc: dict | None = None,
    top_n: int | None = None,
) -> dict:
    """
    Rerank by overall score and write outputs to data/runs/<ts>/final.json + report.md.
    Pulls missing pieces from the in-memory session.
    """
    # Defaults
    if top_n is None:
        top_n = SETTINGS.SHOW_TOP

    # Pull from session if not provided
    if job_json is None:
        job_json = get_job()
    if shortlisted is None:
        shortlisted = get_shortlist()
    if qa_by_doc is None:
        qa_by_doc = get_qa_map()
    if scores_by_doc is None:
        scores_by_doc = get_score_map()

    # Guardrails
    if not job_json:
        raise ValueError("finalize_and_save: job_json is empty. Run parse_job first.")
    if not shortlisted:
        raise ValueError("finalize_and_save: shortlist is empty. Run retrieve_candidates first.")

    # Build strong-typed objects
    job = JobListing(**job_json)

    entries: list[dict] = []
    for row in shortlisted:
        cand = CandidateProfile(**row["candidate"])
        doc = cand.doc_id

        qa_src = qa_by_doc.get(doc, [])
        qa_list = [QAItem(**q) for q in qa_src]

        sc_src = scores_by_doc.get(doc)
        if sc_src is None:
            # Safe default if someone slipped through without scoring
            sc_obj = ScoreBreakdown(
                skills_score=0,
                experience_score=0,
                education_score=0,
                overall_score=0,
                reasons={},
            )
        else:
            sc_obj = ScoreBreakdown(**sc_src)

        entries.append({
            "doc_id": doc,
            "vector_score": row.get("score", 0.0),
            "candidate": cand,        # NOTE: keep as CandidateProfile object
            "qa": qa_list,            # list[QAItem]
            "scores": sc_obj,         # ScoreBreakdown object (required by rerank_top)
        })

    # Rerank + persist
    top = rerank_top(entries, top_n=top_n)
    paths = save_outputs(job, top)
    return paths


@tool("run_pipeline", return_direct=True)
def lc_run_pipeline(raw_text: str) -> dict:
    """
    End-to-end pipeline in one shot (token-efficient):
      1) parse job
      2) retrieve candidates (SETTINGS.TOP_K)
      3) for each candidate: generate N questions (SETTINGS.Q_MIN..Q_MAX), simulate answers, score
      4) rerank top SHOW_TOP and save final outputs
    Returns: {"final_json_path": "...", "report_md_path": "..."}
    """
    # 1) Parse
    job = parse_job_to_json(raw_text)
    job_json = job.model_dump()
    set_job(job_json)

    # 2) Retrieve
    hits = search_candidates(job, top_k=SETTINGS.TOP_K)
    shortlist = []
    for h in hits:
        cand: CandidateProfile = h["candidate"]
        shortlist.append({"id": h["id"], "score": h["score"], "candidate": cand.model_dump()})
    set_shortlist(shortlist)

    # 3) Per-candidate Q&A + score (stay out of LLM context; store in session)
    qa_by_doc: dict[str, list[dict]] = {}
    scores_by_doc: dict[str, dict] = {}

    for row in shortlist:
        cand = CandidateProfile(**row["candidate"])
        # 3a) questions
        qs = generate_questions(job, cand, n_min=SETTINGS.Q_MIN, n_max=SETTINGS.Q_MAX)
        qa_items = [q.model_dump() for q in qs]
        set_qa_for(cand.doc_id, qa_items)
        qa_by_doc[cand.doc_id] = qa_items

        # 3b) simulate answers
        answered = simulate_answers(job, cand, [QAItem(**q) for q in qa_items])
        answered_items = [a.model_dump() for a in answered]
        set_qa_for(cand.doc_id, answered_items)   # overwrite with answers
        qa_by_doc[cand.doc_id] = answered_items

        # 3c) score candidate
        sc = score_candidate(job, cand, [QAItem(**a) for a in answered_items])
        scores_by_doc[cand.doc_id] = sc.model_dump()
        set_score_for(cand.doc_id, sc.model_dump())

    # 4) Rerank + save
    # Build the structure rerank/save expect
    compiled = []
    for row in shortlist:
        cand = CandidateProfile(**row["candidate"])
        doc = cand.doc_id
        qa_list = [QAItem(**x) for x in qa_by_doc.get(doc, [])]
        sb = ScoreBreakdown(**scores_by_doc.get(doc, {
            "skills_score": 0, "experience_score": 0, "education_score": 0, "overall_score": 0, "reasons": {}
        }))
        compiled.append({
            "doc_id": doc,
            "vector_score": row["score"],
            "candidate": cand,
            "qa": qa_list,
            "scores": sb,
        })

    top = rerank_top(compiled, top_n=SETTINGS.SHOW_TOP)
    paths = save_outputs(job, top)

    # keep session mirrors up to date
    set_qa_map(qa_by_doc)
    set_score_map(scores_by_doc)

    return paths