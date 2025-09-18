from __future__ import annotations
from typing import List, Dict, Any, Tuple

from agent.clients import get_clients
from agent.utils import append_usage
from agent.config import SETTINGS

def chat_json_with_logging(
    messages: List[Dict[str, str]],
    model: str,
    tag: str,
    **kwargs
) -> Tuple[str, Any]:
    """
    Calls chat.completions and logs usage. Returns (content, full_response).
    - Logs prompt_tokens, completion_tokens, total_tokens to tokens_count/*
    - Assumes JSON response (set response_format in kwargs if needed)
    """
    c = get_clients()
    resp = c.openai.chat.completions.create(model=model, messages=messages, **kwargs)
    usage = getattr(resp, "usage", None) or {}
    # Azure sometimes uses input/output_tokens; OpenAI uses prompt/completion_tokens
    append_usage(
        kind="chat",
        model=model,
        tag=tag,
        prompt_tokens=getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
        completion_tokens=getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
        total_tokens=getattr(usage, "total_tokens", None),
    )
    content = (resp.choices[0].message.content or "").strip()
    return content, resp

def embed_with_logging(texts: List[str], model: str, tag: str) -> List[List[float]]:
    """
    Uses the OpenAI client for embeddings so usage is available and logged.
    Returns a list of vectors aligned with 'texts'.
    """
    c = get_clients()
    resp = c.openai.embeddings.create(model=model, input=texts)
    usage = getattr(resp, "usage", None) or {}
    append_usage(
        kind="embedding",
        model=model,
        tag=tag,
        prompt_tokens=getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
        completion_tokens=0,
        total_tokens=getattr(usage, "total_tokens", None),
    )
    return [d.embedding for d in resp.data]