import logging

from .config import DISEASE_KEYWORDS
from .core.asr import transcribe_audio
from .core.clip_validator import is_plant_leaf
from .core.hybrid import generate_response_hybrid
from .core.tts import synthesize_audio
from .core.vision import (
    detect_disease_from_image,
    get_image_reliability_metrics,
    is_valid_leaf_image,
)

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

        if not is_valid_leaf_image(probs):
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

    rag_result = generate_response_hybrid(question, target_lang=target_lang, k=k)
    output["answer_text"] = rag_result["answer"]
    output["sources"] = rag_result["sources"]

    if return_audio and rag_result["answer"]:
        output["audio_path"] = synthesize_audio(rag_result["answer"], lang=target_lang)

    return output
