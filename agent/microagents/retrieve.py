from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from agent.clients import get_clients
from agent.config import SETTINGS
from agent.llm import embed_with_logging
from agent.schemas import CandidateProfile, JobListing


def build_job_fusion(job: JobListing) -> str:
    """Compact fusion string used for retrieval embeddings."""
    skills = ", ".join(job.required_skills or [])
    return (
        f"title: {job.title or ''}\n"
        f"skills: {skills}\n"
        f"experience_required: {job.experience_required or ''}\n"
        f"education_required: {job.education_required or ''}"
    )

def search_candidates(job: JobListing, top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    - Embeds the job fusion text with your Azure deployment (team12-embedding)
    - Logs embedding tokens
    - Queries Qdrant and returns a list of {id, score, candidate} (CandidateProfile)
    """
    fusion = build_job_fusion(job)
    vec = embed_with_logging([fusion], model=SETTINGS.EMBEDDING_DEPLOYMENT, tag="retrieve_job")[0]

    clients = get_clients()
    hits = clients.qdrant.search(
        collection_name=SETTINGS.QDRANT_COLLECTION,
        query_vector=vec,
        limit=top_k or SETTINGS.TOP_K,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        try:
            cand = CandidateProfile(**payload)
        except Exception:
            # If payload has unexpected fields, fall back to minimal mapping
            cand = CandidateProfile(
                doc_id=payload.get("doc_id", str(h.id)),
                name=payload.get("name"),
                contact=payload.get("contact") or {},
                skills=payload.get("skills") or [],
                experience=payload.get("experience") or [],
                education=payload.get("education") or [],
                certifications=payload.get("certifications") or [],
                raw_text=payload.get("raw_text"),
                embed_meta=payload.get("embed_meta") or {},
                _orig_id=payload.get("_orig_id"),
            )
        results.append({"id": str(h.id), "score": float(h.score), "candidate": cand})
    return results