from typing import Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    text_question: Optional[str] = None
    target_lang: Literal["ha", "ff", "fr", "en"] = "fr"
    return_audio: bool = True
    k: int = Field(default=2, ge=1, le=10)


class Source(BaseModel):
    score: float
    title: str
    lang: str


class AskResponse(BaseModel):
    detected_disease: Optional[str] = None
    question: str
    target_lang: Literal["ha", "ff", "fr", "en"]
    answer_text: str
    sources: list[Source]
    audio_url: Optional[str] = None
    intent: Optional[str] = None
