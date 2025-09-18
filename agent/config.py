from dataclasses import dataclass
import os

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

@dataclass(frozen=True)
class Settings:
    # Qdrant 
    QDRANT_URL: str = "https://5b09d271-e95a-480d-a2db-8dc806bcc952.us-east-1-1.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY: str | None = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.JC3qRCgKASRAxlWcEryGeUcdSRf_cRCb4oc1WsOJk9s"
    QDRANT_COLLECTION: str = "recruitflow_resumes_v1"

    # Embedding deployment 
    EMBEDDING_DEPLOYMENT: str = "team12-embedding"
    EMBEDDING_DIM: int = 1536

        # --- LLAMA (Answer Simulation) ---
    # Using separate Azure resource + deployment for Llama-4-Maverick-17B-128E-Instruct-FP8
    LLAMA_BASE_URL: str = "https://aoai-recruitflow-resource.services.ai.azure.com"
    LLAMA_API_VERSION: str = "2024-05-01-preview"
    LLAMA_DEPLOYMENT: str = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    LLAMA_API_KEY: str = "API_KEY"   

    # GPT model for scoring
    CHAT_DEPLOYMENT: str = "team12-gpt4o"

    # Retrieval params (adjust top_k as needed - take into account the cost of downstream steps)
    TOP_K: int = 10
    SHOW_TOP: int = TOP_K
    Q_MIN: int = 4
    Q_MAX: int = 5

SETTINGS = Settings()
