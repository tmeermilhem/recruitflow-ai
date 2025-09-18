from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class JobListing(BaseModel):
    title: str
    required_skills: List[str] = Field(default_factory=list)
    experience_required: Optional[str] = None
    education_required: Optional[str] = None
    extra_attributes: List[str] = Field(default_factory=list)

class ExperienceItem(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)

class EducationItem(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    school: Optional[str] = None
    year: Optional[str] = None

class Contact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None

class CandidateProfile(BaseModel):
    doc_id: str
    source_path: Optional[str] = None
    name: Optional[str] = None
    contact: Contact = Field(default_factory=Contact)
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    embed_meta: Dict[str, Any] = Field(default_factory=dict)
    _orig_id: Optional[str] = None

class QAItem(BaseModel):
    question: str
    answer: str

class ScoreBreakdown(BaseModel):
    skills_score: int
    experience_score: int
    education_score: int
    overall_score: int
    reasons: Dict[str, str] = Field(default_factory=dict)

class RunArtifacts(BaseModel):
    run_id: str
    job: JobListing
    shortlist: List[CandidateProfile] = Field(default_factory=list)
    qa_pairs: Dict[str, List[QAItem]] = Field(default_factory=dict)  # key: doc_id
    scores: Dict[str, ScoreBreakdown] = Field(default_factory=dict)   # key: doc_id
