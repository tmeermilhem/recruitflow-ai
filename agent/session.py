# agent/session.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

# Simple in-memory session (reset each process run)

_STATE: Dict[str, Any] = {
    # Global context
    "job": None,                 # dict (JobListing)
    "shortlist": None,           # list of {'id','score','candidate': dict}

    # Per-candidate aggregates
    "qa_by_doc": {},             # dict[str, list[{'question','answer'}]]
    "scores_by_doc": {},         # dict[str, dict]

    # Working-set (the candidate the agent is currently handling)
    "current_doc_id": None,      # str
    "current_candidate": None,   # dict (CandidateProfile)
    "current_questions": None,   # list[dict] (QAItem-like: {'question','answer'?})
    "current_score": None,       # dict (ScoreBreakdown)
}

# ---- Job ----
def set_job(job: Dict[str, Any]) -> None:
    _STATE["job"] = job

def get_job() -> Optional[Dict[str, Any]]:
    return _STATE.get("job")

# ---- Shortlist ----
def set_shortlist(shortlist: List[Dict[str, Any]]) -> None:
    _STATE["shortlist"] = shortlist

def get_shortlist() -> Optional[List[Dict[str, Any]]]:
    return _STATE.get("shortlist")

# ---- QA Map ----
def set_qa_for(doc_id: str, qa_list: List[Dict[str, Any]]) -> None:
    _STATE["qa_by_doc"][doc_id] = qa_list

def get_qa_for(doc_id: str) -> List[Dict[str, Any]]:
    return _STATE["qa_by_doc"].get(doc_id, [])

def get_qa_map() -> Dict[str, List[Dict[str, Any]]]:
    return _STATE["qa_by_doc"]

# ---- Scores Map ----
def set_score_for(doc_id: str, score: Dict[str, Any]) -> None:
    _STATE["scores_by_doc"][doc_id] = score

def get_score_for(doc_id: str) -> Optional[Dict[str, Any]]:
    return _STATE["scores_by_doc"].get(doc_id)

def get_score_map() -> Dict[str, Dict[str, Any]]:
    return _STATE["scores_by_doc"]

# ---- Working-set (current candidate) ----
def set_current_doc_id(doc_id: Optional[str]) -> None:
    _STATE["current_doc_id"] = doc_id

def get_current_doc_id() -> Optional[str]:
    return _STATE.get("current_doc_id")

def set_current_candidate(candidate: Optional[Dict[str, Any]]) -> None:
    _STATE["current_candidate"] = candidate

def get_current_candidate() -> Optional[Dict[str, Any]]:
    return _STATE.get("current_candidate")

def set_current_questions(questions: Optional[List[Dict[str, Any]]]) -> None:
    _STATE["current_questions"] = questions

def get_current_questions() -> Optional[List[Dict[str, Any]]]:
    return _STATE.get("current_questions")

def set_current_score(score: Optional[Dict[str, Any]]) -> None:
    _STATE["current_score"] = score

def get_current_score() -> Optional[Dict[str, Any]]:
    return _STATE.get("current_score")

def set_qa_map(full_map: Dict[str, List[Dict[str, Any]]]) -> None:
    _STATE["qa_by_doc"] = full_map

def set_score_map(full_map: Dict[str, Dict[str, Any]]) -> None:
    _STATE["scores_by_doc"] = full_map

# ---- Reset helpers ----
def reset_session() -> None:
    _STATE["job"] = None
    _STATE["shortlist"] = None
    _STATE["qa_by_doc"] = {}
    _STATE["scores_by_doc"] = {}
    _STATE["current_doc_id"] = None
    _STATE["current_candidate"] = None
    _STATE["current_questions"] = None
    _STATE["current_score"] = None