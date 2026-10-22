import json
import logging

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("RAG-Gateway")

GROQ_DIRECT_URL = "https://api.groq.com/openai/v1"

# Fall back to the next provider on rate limits, timeouts and provider-side errors.
FALLBACK_STATUS_CODES = [408, 429, 500, 502, 503, 504]


def portkey_enabled() -> bool:
    return bool(settings.PORTKEY_API_KEY and (settings.PORTKEY_CONFIG_ID or settings.PORTKEY_PROVIDER_PRIMARY))


def _routing_config() -> str:
    if settings.PORTKEY_CONFIG_ID:
        return settings.PORTKEY_CONFIG_ID

    # Inline config; rejected by workspaces that enable block_inline_config.
    providers = [p for p in (settings.PORTKEY_PROVIDER_PRIMARY, settings.PORTKEY_PROVIDER_FALLBACK) if p]
    if len(providers) == 1:
        return json.dumps({"provider": providers[0]})
    return json.dumps({
        "strategy": {"mode": "fallback", "on_status_codes": FALLBACK_STATUS_CODES},
        "targets": [{"provider": p} for p in providers],
    })


def _portkey_headers(component: str) -> dict:
    return {
        "x-portkey-api-key": settings.PORTKEY_API_KEY,
        "x-portkey-config": _routing_config(),
        "x-portkey-metadata": json.dumps({"app": "compute-match-rag", "component": component}),
    }


def get_openai_client_params(component: str) -> dict:
    """
    Connection settings for any OpenAI-compatible client (base_url, api_key, default_headers).

    Routes through the Portkey gateway when it is configured, otherwise calls Groq directly.
    """
    if portkey_enabled():
        # Provider credentials live in Portkey's Model Catalog, so the OpenAI api_key is unused.
        return {
            "base_url": settings.PORTKEY_GATEWAY_URL,
            "api_key": "portkey",
            "default_headers": _portkey_headers(component),
        }

    logger.warning(f"Portkey gateway not configured; '{component}' is calling Groq directly.")
    return {"base_url": GROQ_DIRECT_URL, "api_key": settings.GROQ_API_KEY}


def get_chat_llm(component: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=settings.LLM_MODEL, temperature=temperature, **get_openai_client_params(component))
