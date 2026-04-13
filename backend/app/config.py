from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # ML Model paths
    E5_MODEL_NAME: str = "intfloat/e5-large"
    BART_MODEL_NAME: str = "facebook/bart-large-mnli"
    FAISS_INDEX_DIR: str = "./data/faiss_index"
    MODEL_CACHE_DIR: str = "./data/models"

    # ML Hyperparameters (from paper)
    FAISS_K_NEIGHBOURS: int = 10
    SEMANTIC_CONF_THRESHOLD: float = 0.78
    SEMANTIC_COSINE_THRESHOLD: float = 0.85
    HDBSCAN_MIN_CLUSTER_SIZE: int = 3
    HDBSCAN_MIN_SAMPLES: int = 2
    RECURRENCE_ALPHA: float = 0.55
    GATE_SEMANTIC_HIGH: float = 0.90
    GATE_BEHAVIOURAL_HIGH: float = 0.90
    COLD_START_THRESHOLD: int = 10
    STAGE2_NEIGHBOUR_K: int = 5
    STAGE2_AGREEMENT_THRESHOLD: float = 0.70
    STAGE2_DISTANCE_THRESHOLD: float = 0.35
    BART_ENTAILMENT_THRESHOLD: float = 0.70
    MANUAL_REVIEW_FLOOR: float = 0.0

    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
