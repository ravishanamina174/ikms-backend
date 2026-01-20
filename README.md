# IKMS — Intelligent Knowledge & Multi-Agent System (IKMS)

Brief, practical README covering architecture, setup, and usage for the IKMS project.

**Overview**
- **Purpose:** Question-answering over indexed documents using a multi-agent retrieval-augmented generation (RAG) pipeline.
- **Frontend:** Hosted at `https://ikms-lake.vercel.app` (UI for asking questions and uploading PDFs).
- **Backend:** API and services that run the RAG agents. Repo: `https://github.com/ravishanamina174/ikms-backend` — live at `https://ikms-backend-6655.onrender.com`.

**Architecture**
- **API layer:** `backend/src/app/api.py` — exposes `/qa` and `/index-pdf` endpoints.
- **Services:** `backend/src/app/services/` — `indexing_service.py` (PDF ingest & vector indexing) and `qa_service.py` (multi-agent orchestration).
- **Core agents & tools:** `backend/src/app/core/agents/` — agent graph, prompts, tools, and state management.
- **LLM Factory:** `backend/src/app/core/llm/factory.py` — abstracts LLM client creation.
- **Retrieval:** `backend/src/app/core/retrieval/` — vector store + serialization helpers.

Setup (local development)
- Prereqs: `python 3.10+`, `pip`, optional `virtualenv`.
- From repository root run:

```bash
# change to backend
cd backend

# create and activate virtualenv (macOS)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables (example)
export OPENAI_API_KEY="sk_your_key_here"
export FRONTEND_URL="https://ikms-lake.vercel.app"

# Run the API (from backend directory)
uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- The `OPENAI_API_KEY` is used by the LLM factory. If missing, the `/qa` endpoint returns a friendly placeholder.
- `FRONTEND_URL` is configured by default to `https://ikms-lake.vercel.app` in `api.py`.

Quick API usage
- Index a PDF (multipart form):

```bash
curl -F "file=@path/to/doc.pdf" http://localhost:8000/index-pdf
```

- Ask a question (JSON):

```bash
curl -X POST http://localhost:8000/qa \
	-H "Content-Type: application/json" \
	-d '{"question": "What is RAG?", "use_planning": false}'
```

Code documentation (quick map)
- `backend/src/app/api.py`: FastAPI application and endpoints.
- `backend/src/app/services/indexing_service.py`: PDF ingestion and vector indexing helpers.
- `backend/src/app/services/qa_service.py`: High-level QA orchestration; calls into the agent graph.
- `backend/src/app/core/agents/agents.py` and `graph.py`: agent definitions and orchestration logic.
- `backend/src/app/core/llm/factory.py`: LLM client creation and configuration.
- `backend/src/app/core/retrieval/vector_store.py` and `serialization.py`: vector store integration and persistence.

User guide / How to use
- Upload PDFs via the `/index-pdf` endpoint to add source content to the vector store.
- Use the `/qa` endpoint to ask natural-language questions. The response contains `answer`, optional `plan`, and `context` fields.
- For quick testing use the provided live backend at `https://ikms-backend-6655.onrender.com` and the frontend at `https://ikms-lake.vercel.app`.

Deploy & env
- The backend is deployed at `https://ikms-backend-6655.onrender.com` (see repo above for deployment details).
- Ensure `OPENAI_API_KEY` (or other LLM credentials) are configured in production environment variables.

Contributing
- Follow standard practice: create feature branches, run tests (if any), and open PRs against `main`.

Contact
- Repo owner: `ravishanamina174` (see GitHub repo link above).

