import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import AskRequest, AskResponse
from .rag import answer_question, index_document


settings = get_settings()

app = FastAPI(
    title="ResearchOS API",
    version="0.1.0",
    description="Multi-document RAG research assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024


def safe_filename(filename: str) -> str:
    clean_name = Path(filename).name
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_name)

    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    return clean_name


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ResearchOS API is running.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "researchos-api",
    }


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename missing.",
        )

    filename = safe_filename(file.filename)

    if file.content_type not in {
        "application/pdf",
        "application/x-pdf",
    }:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have PDF content type.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File size limit is "
                f"{settings.max_file_size_mb} MB."
            ),
        )

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF.",
        )

    stored_filename = f"{uuid.uuid4().hex}_{filename}"
    file_path = Path(settings.upload_dir) / stored_filename
    file_path.write_bytes(content)

    try:
        result = index_document(
            file_path=str(file_path),
            document_name=filename,
        )
    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=f"Document processing failed: {str(error)}",
        ) from error

    return result


@app.post(
    "/api/research/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
) -> AskResponse:
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            document_hash=request.document_hash,
        )

        return AskResponse(**result)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Research request failed: {str(error)}",
        ) from error