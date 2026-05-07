from .config import DISEASE_KEYWORDS
from .core.asr import transcribe_audio
from .core.hybrid import generate_response_hybrid
from .core.tts import synthesize_audio
from .core.vision import detect_disease_from_image


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
        disease, _confidence = detect_disease_from_image(image_path)
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
