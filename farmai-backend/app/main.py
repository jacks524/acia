from pathlib import Path
import shutil
from typing import Literal
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import AUDIO_OUTPUT_DIR, CORS_ORIGINS
from .pipeline import farmai_assistant
from .schemas import AskRequest, AskResponse

tags_metadata = [
    {
        "name": "system",
        "description": "Endpoints de santé et de découverte de l'API.",
    },
    {
        "name": "assistant",
        "description": "Questions texte, image ou audio vers l'assistant FarmAI.",
    },
]

app = FastAPI(
    title="FarmAI API",
    version="1.0.0",
    description=(
        "API quadrilingue FarmAI Cameroun pour conseiller les producteurs de tomate "
        "en français, anglais, Hausa et Fulfulde/Fulfulbe."
    ),
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=str(AUDIO_OUTPUT_DIR)), name="audio")

UPLOAD_DIR = Path("/tmp/farmai_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", tags=["system"])
def root():
    return {
        "name": "FarmAI API",
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.post("/ask/text", response_model=AskResponse, tags=["assistant"])
def ask_text(req: AskRequest):
    if not req.text_question:
        raise HTTPException(status_code=400, detail="text_question requis")

    result = farmai_assistant(
        text_question=req.text_question,
        target_lang=req.target_lang,
        return_audio=req.return_audio,
        k=req.k,
    )
    return _format_response(result)


@app.post("/ask/image", response_model=AskResponse, tags=["assistant"])
async def ask_image(
    image: UploadFile = File(...),
    target_lang: Literal["ha", "ff", "fr", "en"] = Form("fr"),
    return_audio: bool = Form(True),
    k: int = Form(2, ge=1, le=10),
):
    suffix = Path(image.filename or "").suffix
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = farmai_assistant(
            image_path=str(temp_path),
            target_lang=target_lang,
            return_audio=return_audio,
            k=k,
        )
        return _format_response(result)
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/ask/audio", response_model=AskResponse, tags=["assistant"])
async def ask_audio(
    audio: UploadFile = File(...),
    target_lang: Literal["ha", "ff", "fr", "en"] = Form("ha"),
    return_audio: bool = Form(True),
    k: int = Form(2, ge=1, le=10),
):
    suffix = Path(audio.filename or "").suffix
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        result = farmai_assistant(
            audio_question_path=str(temp_path),
            target_lang=target_lang,
            return_audio=return_audio,
            k=k,
        )
        return _format_response(result)
    finally:
        temp_path.unlink(missing_ok=True)


def _format_response(result: dict) -> AskResponse:
    if result.get("audio_path"):
        filename = Path(result["audio_path"]).name
        result["audio_url"] = f"/audio/{filename}"

    payload = {key: value for key, value in result.items() if key != "audio_path"}
    return AskResponse(**payload)
