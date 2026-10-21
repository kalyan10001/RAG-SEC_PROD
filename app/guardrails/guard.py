import os
import logging
from functools import lru_cache

from nemoguardrails import LLMRails, RailsConfig

logger = logging.getLogger("RAG-Guardrails")

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "guardrails")


@lru_cache(maxsize=1)
def get_guardrails() -> LLMRails:
    logger.info(f"Loading NeMo Guardrails configuration from: {CONFIG_DIR}")
    return LLMRails(RailsConfig.from_path(CONFIG_DIR))


def check_query_with_guardrails(query_text: str) -> tuple[bool, str | None]:
    """
    Runs only the input rails defined in guardrails/config.yml against the query.

    Returns (True, None) if the query is allowed, or (False, refusal_message) if a rail blocked it.
    """
    try:
        result = get_guardrails().generate(
            messages=[{"role": "user", "content": query_text}],
            options={"rails": ["input"], "log": {"activated_rails": True}},
        )
    except Exception as e:
        # Fail open so an LLM outage or rate limit on the guard does not take the API down.
        logger.error(f"NeMo Guardrails check failed, allowing query: {e}")
        return True, None

    blocked_by = [rail.name for rail in result.log.activated_rails if rail.stop]
    if blocked_by:
        refusal = result.response[0]["content"]
        logger.warning(f"Query blocked by guardrails {blocked_by}: '{query_text}'")
        return False, refusal

    return True, None
