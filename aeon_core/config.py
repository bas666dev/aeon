"""Configuration primitives shared across Aeon services."""

from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    mongo_uri: str = Field("mongodb://localhost:27017/aeon", env="MONGO_URI")
    mongo_db: str = Field("aeon", env="MONGO_DB")
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("neo4jpass", env="NEO4J_PASS")
    hybrid_search_top_k: int = Field(20, env="HYBRID_SEARCH_TOP_K")
    maat_min_score: float = Field(0.8, env="MAAT_MIN_SCORE")
    default_visibility: str = Field("private", env="DEFAULT_VISIBILITY")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def load_settings() -> Settings:
    """Load a cached copy of the application settings."""

    return Settings()  # type: ignore[arg-type]


settings = load_settings()

__all__ = ["Settings", "settings", "load_settings"]
