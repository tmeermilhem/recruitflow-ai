# agent/clients.py
from __future__ import annotations
from dataclasses import dataclass

from openai import AzureOpenAI, OpenAI
from qdrant_client import QdrantClient

from agent.config import SETTINGS, API_KEY

# ===== Azure OpenAI (GPT-4o + embeddings) =====
AZURE_BASE_URL: str = "https://096290-oai.openai.azure.com"   
API_VERSION: str = "2023-05-15"                                


@dataclass(frozen=True)
class Clients:
    """Primary clients used across the project."""
    openai: AzureOpenAI
    qdrant: QdrantClient


def get_clients() -> Clients:
    """
    Returns:
      - AzureOpenAI client configured for your GPT-4o & embeddings deployments
      - Qdrant client for vector DB
    """
    if not API_KEY:
        raise RuntimeError("API_KEY missing. Put it in .env as API_KEY=<your Azure OpenAI key>")

    # Azure OpenAI client for chat + embeddings on your GPT-4o resource
    oa = AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_BASE_URL,   # NOTE: do NOT append '/openai'
    )

    # Qdrant client
    qc = QdrantClient(
        url=SETTINGS.QDRANT_URL,
        api_key=SETTINGS.QDRANT_API_KEY
    )

    return Clients(openai=oa, qdrant=qc)


# ===== LLAMA client (Azure AI Inference: services.ai.azure.com) =====
from openai import OpenAI

def get_llama_client() -> OpenAI:
    if not SETTINGS.LLAMA_API_KEY:
        raise RuntimeError("LLAMA_API_KEY missing in config.py")

    llama = OpenAI(
        # point at /models so SDK's "/chat/completions" becomes "/models/chat/completions"
        base_url=f"{SETTINGS.LLAMA_BASE_URL}/models",
        default_query={"api-version": SETTINGS.LLAMA_API_VERSION},

        # send key both ways to satisfy services.ai
        api_key=SETTINGS.LLAMA_API_KEY,
        default_headers={"api-key": SETTINGS.LLAMA_API_KEY},
    )
    return llama