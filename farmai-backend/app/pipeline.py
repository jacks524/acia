import logging
import math

from .config import (
    CLIP_ALWAYS_VALIDATE,
    CONFIDENCE_THRESHOLD,
    DISEASE_KEYWORDS,
    ENTROPY_THRESHOLD,
    USE_CLIP_VALIDATOR,
)
from .core.intent import INTENT_DISEASE_QUERY, classify_intent
from .core.responses import build_response

logger = logging.getLogger(__name__)


class ImageNotRecognizedError(ValueError):
    def __init__(self, max_confidence: float, entropy: float):
        self.payload = {
            "error": "image_not_recognized",
            "error_code": 422,
            "message_fr": (
                "L'image ne semble pas être une feuille de tomate. "
                "Veuillez prendre une photo claire d'une feuille."
            ),
            "message_en": (
                "The image does not appear to be a tomato leaf. "
                "Please take a clear photo of a leaf."
            ),
            "message_ha": (
                "Hoton ba kamar ganyen tumatur ba ne. "
                "Da fatan za ka dauki hoto bayyananne na ganye."
            ),
            "message_ff": (
                "Natal ngal nanndaani e haako tomati. "
                "Yiɗde ƴetto natal laaɓɗo no haako waɗi."
            ),
            "max_confidence": max_confidence,
            "entropy": entropy,
        }
        super().__init__(self.payload["message_en"])


def transcribe_audio(audio_path: str, lang: str = "ha") -> str:
    from .core.asr import transcribe_audio as _transcribe_audio

    return _transcribe_audio(audio_path, lang=lang)


def is_plant_leaf(image_path: str) -> bool:
    from .core.clip_validator import is_plant_leaf as _is_plant_leaf

    return _is_plant_leaf(image_path)


def generate_response_hybrid(question: str, target_lang: str = "fr", k: int = 2) -> dict:
    from .core.hybrid import generate_response_hybrid as _generate_response_hybrid

    return _generate_response_hybrid(question, target_lang=target_lang, k=k)


def synthesize_audio(text: str, lang: str = "fr") -> str:
    from .core.tts import synthesize_audio as _synthesize_audio

    return _synthesize_audio(text, lang=lang)


def detect_disease_from_image(image_path: str):
    from .core.vision import detect_disease_from_image as _detect_disease_from_image

    return _detect_disease_from_image(image_path)


def get_image_reliability_metrics(probs):
    values = [max(float(prob), 1e-8) for prob in probs]
    total = sum(values) or 1.0
    values = [prob / total for prob in values]
    entropy = -sum(prob * math.log(prob) for prob in values)
    return {
        "max_confidence": max(values),
        "entropy": entropy / math.log(len(values)),
    }


def is_valid_leaf_image(probs) -> bool:
    metrics = get_image_reliability_metrics(probs)
    return (
        metrics["max_confidence"] >= CONFIDENCE_THRESHOLD
        and metrics["entropy"] <= ENTROPY_THRESHOLD
    )


def farmai_assistant(
    image_path=None,
    text_question=None,
    audio_question_path=None,
    target_lang="fr",
    return_audio=True,
    k=2,
):
    output = {
        "detected_disease": None,
        "question": None,
        "target_lang": target_lang,
        "answer_text": None,
        "sources": [],
        "audio_path": None,
    }

    if image_path:
        disease, _confidence, probs = detect_disease_from_image(image_path)
        metrics = get_image_reliability_metrics(probs)
        level_1_valid = is_valid_leaf_image(probs)

        if USE_CLIP_VALIDATOR and (CLIP_ALWAYS_VALIDATE or not level_1_valid):
            clip_check = is_plant_leaf(image_path)
            if not clip_check:
                logger.info(
                    "Image rejected: max_conf=%.2f, entropy=%.2f, CLIP_check=%s",
                    metrics["max_confidence"],
                    metrics["entropy"],
                    clip_check,
                )
                raise ImageNotRecognizedError(
                    max_confidence=metrics["max_confidence"],
                    entropy=metrics["entropy"],
                )
        elif not level_1_valid:
            logger.info(
                "Image rejected: max_conf=%.2f, entropy=%.2f, CLIP_check=%s",
                metrics["max_confidence"],
                metrics["entropy"],
                False,
            )
            raise ImageNotRecognizedError(
                max_confidence=metrics["max_confidence"],
                entropy=metrics["entropy"],
            )

        output["detected_disease"] = disease
        keyword = DISEASE_KEYWORDS.get(disease, disease)
        question = f"Quels conseils pour {keyword} sur mes tomates ?"
    elif audio_question_path:
        question = transcribe_audio(audio_question_path, lang=target_lang)
    elif text_question:
        question = text_question
    else:
        raise ValueError("Fournir au moins image, texte ou audio.")

    output["question"] = question

    intent = classify_intent(question, target_lang)
    if intent != INTENT_DISEASE_QUERY:
        output = build_response(intent, target_lang, original_question=question)
        if return_audio and output["answer_text"]:
            output["audio_path"] = synthesize_audio(output["answer_text"], lang=target_lang)
        return output

    rag_result = generate_response_hybrid(question, target_lang=target_lang, k=k)
    output["answer_text"] = rag_result["answer"]
    output["sources"] = rag_result["sources"]
    output["intent"] = INTENT_DISEASE_QUERY

    if return_audio and rag_result["answer"]:
        output["audio_path"] = synthesize_audio(rag_result["answer"], lang=target_lang)

    return output
