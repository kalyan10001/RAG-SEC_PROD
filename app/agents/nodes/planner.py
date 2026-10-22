from app.agents.state import AgentState
from app.gateway import get_chat_llm

llm = get_chat_llm("planner", temperature=0)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are the query planner for the Compute-Match Enterprise AI Assistant.
    Compute-Match is NVIDIA's GPU capacity marketplace connecting customers with NVIDIA Cloud Providers (NCPs).
    The knowledge base covers: sign-in and onboarding, roles and access, customer and NCP guides,
    reservation request lifecycle and statuses, capacity and CSV import, notifications and emails,
    system architecture, backend APIs, Temporal workflows, Keycloak identity, analytics,
    environments and CI, local development, security and observability, Salesforce, and the glossary/FAQ.

    CONVERSATION HISTORY:
    {history}

    LATEST MESSAGE:
    "{user_message}"

    Task:
    1. If the latest message is a greeting or can be answered using ONLY the conversation history
       (e.g. "what is my name"), respond with exactly: CONVERSATIONAL
    2. Otherwise, rewrite it as a concise standalone search query for the knowledge base,
       resolving pronouns and follow-ups using the conversation history.

    Output ONLY 'CONVERSATIONAL' or the search query, with no quotes or explanation.
    """

    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
    decision = content.strip().strip('"')

    if decision.upper() == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    # An empty planner output would make the embedding call fail, so search with the raw message.
    if not decision:
        decision = user_message

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
