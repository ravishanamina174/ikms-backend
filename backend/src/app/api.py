from os import getenv
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from .models import QuestionRequest, QAResponse
from .services.qa_service import answer_question
import openai
from .services.indexing_service import index_pdf_bytes
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Class 12 Multi-Agent RAG Demo",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.1.0",
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# FRONTEND_URL = getenv("FRONTEND_URL", "http://localhost:3000")

FRONTEND_URL = getenv("FRONTEND_URL", "https://ikms-lake.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
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
        # If the error is an OpenAI authentication/authorization issue,
        # return a helpful placeholder response instead of a 500 so the
        # `/qa` endpoint remains reachable for testing (Postman, etc.).
        if getattr(exc, "__class__", None) is not None and (
            exc.__class__.__name__ == "AuthenticationError"
            or "Incorrect API key" in str(exc)
        ):
            return QAResponse(
                answer=(
                    "LLM unavailable: invalid or missing OpenAI API key. "
                    "Configure `OPENAI_API_KEY` in the environment or `.env` file."
                ),
                context="",
            )
        # Re-raise other unexpected exceptions to be handled by the
        # global exception handler.
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

    # Use the bytes-based indexing helper which writes to an OS temp file
    # only if required by the underlying loader, and cleans up after.
    chunks_indexed = index_pdf_bytes(contents, file.filename)

    return {
        "filename": file.filename,
        "chunks_indexed": chunks_indexed,
    }
