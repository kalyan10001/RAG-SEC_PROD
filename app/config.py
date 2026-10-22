import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "Prod-Rag"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_GATEWAY_URL = os.getenv("PORTKEY_GATEWAY_URL", "https://api.portkey.ai/v1")
    # Saved Portkey config ("pc-..."); takes precedence over the provider slugs below
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    # Model Catalog provider slugs, e.g. "@groq-primary" and "@groq-fallback"
    PORTKEY_PROVIDER_PRIMARY = os.getenv("PORTKEY_PROVIDER_PRIMARY")
    PORTKEY_PROVIDER_FALLBACK = os.getenv("PORTKEY_PROVIDER_FALLBACK")


settings = Settings()