# IKMS — Intelligent Knowledge & Multi-Agent System (IKMS)

Brief, practical README covering architecture, setup, and usage for the IKMS project.

**Overview**
- **Purpose:** Question-answering over indexed documents using a multi-agent retrieval-augmented generation (RAG) pipeline.
- **Frontend:** Hosted at `https://ikms-lake.vercel.app` or run locally with `npm run dev`.
- **Backend:** FastAPI app with Gemini LLM and Pinecone vector retrieval.

**Architecture**
- **API layer:** `backend/src/app/api.py` — exposes `/qa` and `/index-pdf` endpoints.
- **Services:** `backend/src/app/services/` — PDF ingest and QA orchestration.
- **Core agents & tools:** `backend/src/app/core/agents/` — agent graph, prompts, tools, and state management.
- **LLM Factory:** `backend/src/app/core/llm/factory.py` — connects to Gemini.
- **Retrieval:** `backend/src/app/core/retrieval/` — Pinecone vector store wrappers.

Setup (local development)
- Prereqs: Python 3.11+, Node 20+.
- Backend:

```bash
cd IKMS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

Set the environment variables in `IKMS/.env`:

```bash
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
GEMINI_API_KEY=your_gemini_key
FRONTEND_URL=http://localhost:3000
FRONTEND_URLS=http://localhost:3000,https://ikms-lake.vercel.app
GEMINI_MODEL_NAME=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL_NAME=models/text-embedding-004
```

Run the API:

```bash
cd IKMS/backend
PYTHONPATH=. uvicorn src.app.api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Quick API usage
- Index a PDF:

```bash
curl -F "file=@path/to/doc.pdf" http://localhost:8000/index-pdf
```

- Ask a question:

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "use_planning": false}'
```

Deployment notes
- Local frontend defaults to `http://localhost:8000` in [frontend/lib/api.ts](frontend/lib/api.ts).
- Set `NEXT_PUBLIC_API_URL` in Vercel to the Render backend URL for deployment.
- On Render, store `GEMINI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` as environment variables.

Contributing
- Use feature branches and keep `.env` local-only.

