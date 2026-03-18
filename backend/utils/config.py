import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM – Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    # Data APIs
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")

    # Vector DB
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "financial_docs")

    # AWS (optional)
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY", "")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY", "")
    AWS_REGION: str = "us-east-1"

    # Agent Settings
    MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "10"))
    REFLECTION_THRESHOLD: float = float(os.getenv("REFLECTION_THRESHOLD", "0.7"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = 384

    # Backtester binary path
    BACKTESTER_BIN: str = os.getenv("BACKTESTER_BIN", "./strategy_engine/bin/backtest")

    # CORS (Fix 5)
    FRONTEND_URL: Optional[str] = os.getenv("FRONTEND_URL", "http://localhost:5173")
    VERCEL_URL: Optional[str] = os.getenv("VERCEL_URL")
