import os
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAG-Frontend")

app = FastAPI(title="Enterprise Agentic RAG Frontend Server")

# Resolve full path to app/index.html
INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")


@app.get("/", response_class=HTMLResponse)
def get_frontend():
    """
    Serves the beautiful Single-Page App chat UI from app/index.html.
    """
    try:
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to read index.html: {e}")
        return HTMLResponse(
            content="<h3>Error loading Web UI. Check console for detail.</h3>", 
            status_code=500
        )
