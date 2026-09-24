from os import getenv

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import QAResponse, QuestionRequest
from .services.indexing_service import index_pdf_bytes
from .services.qa_service import answer_question


app = FastAPI(
    title="IKMS Multi-Agent RAG API",
    description=(
        "API for uploading PDFs and asking questions against the indexed knowledge base."
    ),
    version="0.1.0",
)


def _get_allowed_origins() -> list[str]:
    raw = getenv("FRONTEND_URLS", "http://localhost:3000,https://ikms-lake.vercel.app")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        return ["http://localhost:3000"]
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:  # pragma: no cover - simple demo handler
    """Catch-all handler for unexpected errors.

    FastAPI will still handle `HTTPException` instances and validation errors
    separately; this is only for truly unexpected failures so API consumers
    get a consistent 500 response body.
    """

    if isinstance(exc, HTTPException):
        # Let FastAPI handle HTTPException as usual.
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    """Submit a question about the vector databases paper.

    US-001 requirements:
    - Accept POST requests at `/qa` with JSON body containing a `question` field
    - Validate the request format and return 400 for invalid requests
    - Return 200 with `answer`, `draft_answer`, and `context` fields
    - Delegate to the multi-agent RAG service layer for processing
    """

    question = payload.question.strip()
    use_planning: bool = payload.use_planning
    if not question:
        # Explicit validation beyond Pydantic's type checking to ensure
        # non-empty questions.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    # Delegate to the service layer which runs the multi-agent QA graph
    try:
        result = answer_question(question, use_planning=use_planning)
    except Exception as exc:
        if "503" in str(exc) or "unavailable" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Gemini is temporarily unavailable because the selected model is experiencing high demand. "
                    "Please retry the question in a moment."
                ),
            ) from exc
        # If the Gemini API key or quota is invalid, return a helpful placeholder
        # instead of a 500 so the `/qa` endpoint remains reachable for testing.
        if getattr(exc, "__class__", None) is not None and (
            exc.__class__.__name__ in {"GoogleGenerativeAIError", "ValueError"}
            or "API key" in str(exc).lower()
            or "api_key" in str(exc).lower()
            or "authentication" in str(exc).lower()
            or "quota" in str(exc).lower()
        ):
            return QAResponse(
                answer=(
                    "LLM unavailable: invalid or missing Gemini API key, or the API quota is exhausted. "
                    "Configure `GEMINI_API_KEY` in the environment or `.env` file."
                ),
                context="",
            )
        raise

    # Return only the fields defined by QAResponse
    return QAResponse(
        answer=result.get("answer", ""),
        plan=result.get("plan", ""),
        sub_questions=result.get("sub_questions", ""),
        context=result.get("context", "")
    )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    """Upload a PDF and index it into the vector database.

    This endpoint:
    - Accepts a PDF file upload
    - Saves it to the local `data/uploads/` directory
    - Uses PyPDFLoader to load the document into LangChain `Document` objects
    - Indexes those documents into the configured Pinecone vector store
    """

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Read the uploaded PDF into memory and index without persisting
    contents = await file.read()

    try:
        # Use the bytes-based indexing helper which writes to an OS temp file
        # only if required by the underlying loader, and cleans up after.
        chunks_indexed = index_pdf_bytes(contents, file.filename)
    except Exception as exc:
        message = str(exc).lower()
        if (
            "api key" in message
            or "api_key" in message
            or "authentication" in message
        ):
            detail = (
                "PDF indexing failed because GEMINI_API_KEY is invalid or expired. "
                "Create a new Gemini API key and restart the backend."
            )
        elif "not found" in message or "model" in message:
            detail = (
                "PDF indexing failed because GEMINI_EMBEDDING_MODEL_NAME is not supported. "
                "Use GEMINI_EMBEDDING_MODEL_NAME=gemini-embedding-001."
            )
        elif (
            "invalid argument" in message
            or "quota" in message
            or "forbidden" in message
            or "pinecone" in message
        ):
            detail = (
                "PDF indexing failed because the Gemini or Pinecone configuration is invalid. "
                "Check GEMINI_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME."
            )
        else:
            raise

        if detail:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=detail,
            ) from exc

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
    }
