import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup basic standard python logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG-Application")

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from app.agents.graph import rag_agent
from app.guardrails.guard import check_query_with_guardrails

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI Backend on port 8000
app = FastAPI(title="Enterprise Agentic RAG Backend API")

# Enable Cross-Origin Resource Sharing (CORS) for separate frontend port connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, configure this specifically to your frontend domain (e.g. ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    """
    Returns standard API health details.
    """
    return {
        "status": "online",
        "service": "Enterprise LangGraph RAG Backend API",
        "documentation": "/docs"
    }


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    # Apply NVIDIA NeMo Guardrails
    is_allowed, block_response = check_query_with_guardrails(q)
    if not is_allowed:
        logger.warning(f"Query was blocked by Guardrails: '{q}'")
        return {
            "question": q,
            "answer": block_response,
            "thought_process": ["Request blocked by NVIDIA NeMo Guardrails safety policies."],
            "status": "blocked_by_guardrails",
            "sources": []
        }

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # LangGraph RAG pipeline (Guardrails are removed per user request)
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
        }
    except Exception as e:
        logger.error(f"Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }