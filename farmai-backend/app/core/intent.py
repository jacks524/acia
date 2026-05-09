import re
import shutil
import subprocess
import tempfile
from pathlib import Path

INTENT_GREETING = "INTENT_GREETING"
INTENT_IDENTITY = "INTENT_IDENTITY"
INTENT_THANKS = "INTENT_THANKS"
INTENT_GOODBYE = "INTENT_GOODBYE"
INTENT_DISEASE_QUERY = "INTENT_DISEASE_QUERY"
INTENT_TOMATO_GENERIC = "INTENT_TOMATO_GENERIC"
INTENT_OUT_OF_SCOPE = "INTENT_OUT_OF_SCOPE"
INTENT_UNCLEAR = "INTENT_UNCLEAR"

SEMANTIC_THRESHOLD = 0.5
DISEASE_RETRIEVAL_THRESHOLD = 0.4
UNCLEAR_MIN_CHARS = 3
AUDIO_SILENCE_THRESHOLD_DB = -50.0
AUDIO_MIN_ACTIVE_RATIO = 0.1

GREETINGS = {
    "fr": ["bonjour", "salut", "bonsoir", "coucou", "hello"],
    "en": ["hello", "hi", "good morning", "good evening", "hey"],
    "ha": ["sannu", "barka", "ina kwana", "ina yini", "salam alaikum"],
    "ff": ["jam", "no mbaɗɗaa", "useko", "salaam", "salam"],
}

IDENTITY_KEYWORDS = {
    "fr": ["qui es-tu", "tu es qui", "tu fais quoi", "c'est quoi ton rôle", "ton nom"],
    "en": ["who are you", "what do you do", "your name", "what is your role"],
    "ha": ["kai waye", "ke wacece", "menene aikinka", "sunanka"],
    # TODO: faire valider ces formulations fulfulde par un locuteur natif.
    "ff": ["a woni hombo", "hol no mbaɗata", "inde maa", "ko honɗun golle maa"],
}

THANKS_KEYWORDS = {
    "fr": ["merci", "merci beaucoup", "je te remercie"],
    "en": ["thank you", "thanks", "many thanks"],
    "ha": ["na gode", "nagode", "mun gode"],
    "ff": ["a jaaraama", "useko", "mi yettii"],
}

GOODBYE_KEYWORDS = {
    "fr": ["au revoir", "à bientôt", "a bientot", "bye"],
    "en": ["bye", "goodbye", "see you", "see you later"],
    "ha": ["sai anjima", "sai gobe", "sai wata rana"],
    "ff": ["haa later", "en woodi", "mi yahii"],
}

DISEASE_TERMS = [
    "mildiou",
    "late blight",
    "alternariose",
    "early blight",
    "bacterial spot",
    "tache bactérienne",
    "tache bacterienne",
    "leaf mold",
    "moisissure",
    "septoria",
    "septoriose",
    "spider mites",
    "acariens",
    "target spot",
    "tache cible",
    "mosaic virus",
    "mosaïque",
    "mosaique",
    "yellow leaf curl",
    "enroulement jaune",
    "plante saine",
    "healthy plant",
    "healthy",
    "cuta bakteriya",
    "mildiou tardif",
    "lanƙwashewar ganye",
]

TOMATO_TERMS = [
    "tomate",
    "tomates",
    "tomato",
    "tomatoes",
    "tumatur",
    "tomati",
    "feuille",
    "feuilles",
    "leaf",
    "leaves",
    "ganye",
    "haako",
    "plante",
    "plant",
    "plants",
    "shuka",
    "aawdi",
    *DISEASE_TERMS,
]

GENERIC_TOMATO_TERMS = [
    "arroser",
    "arrosage",
    "water",
    "watering",
    "fertiliser",
    "fertilizer",
    "engrais",
    "compost",
    "planter",
    "plantation",
    "récolter",
    "recolter",
    "harvest",
    "rain",
    "pluie",
    "saison",
    "season",
    "mauvaises herbes",
    "weeds",
    "entretien",
    "entretenir",
    "care",
]

OUT_OF_SCOPE_TERMS = [
    "manioc",
    "cassava",
    "maïs",
    "mais",
    "corn",
    "football",
    "foot",
    "politique",
    "politics",
    "weather forecast",
    "météo demain",
    "meteo demain",
    "temps fera",
]

UNCLEAR_WORD_RE = re.compile(r"^[a-zA-Z]{6,}$")
VOWELS_RE = re.compile(r"[aeiouyàâäéèêëîïôöùûü]", re.IGNORECASE)

PROTOTYPES = {
    INTENT_DISEASE_QUERY: [
        "comment traiter le mildiou de la tomate",
        "my tomato leaves have spots and disease symptoms",
        "tumatur na da tabo a ganye",
        "nyawu haako tomati",
    ],
    INTENT_TOMATO_GENERIC: [
        "comment arroser et entretenir mes tomates",
        "when should I harvest tomato plants",
        "yaya zan kula da tumatur",
        # TODO: faire valider cette phrase fulfulde par un locuteur natif.
        "no mbaawi toppitorde tomati",
    ],
    INTENT_OUT_OF_SCOPE: [
        "quel temps fera-t-il demain",
        "who won the football match",
        "comment cultiver le manioc",
        "political news today",
    ],
}


def classify_intent(text: str | None, lang: str = "fr") -> str:
    """Classe une requête avant RAG."""
    normalized = _normalize(text or "")

    social_intent = _classify_social_intent(normalized, lang)
    if social_intent:
        return social_intent

    if _is_unclear_text(normalized):
        return INTENT_UNCLEAR

    if _contains_any(normalized, OUT_OF_SCOPE_TERMS):
        return INTENT_OUT_OF_SCOPE

    if _contains_any(normalized, DISEASE_TERMS):
        return INTENT_DISEASE_QUERY

    has_tomato_context = _contains_any(normalized, TOMATO_TERMS)
    has_generic_tomato_context = _contains_any(normalized, GENERIC_TOMATO_TERMS)
    if has_tomato_context and has_generic_tomato_context:
        return INTENT_TOMATO_GENERIC

    if has_tomato_context:
        if _has_disease_retrieval_match(normalized, lang):
            return INTENT_DISEASE_QUERY
        return INTENT_TOMATO_GENERIC

    semantic_intent = _classify_semantic_intent(normalized)
    if semantic_intent:
        return semantic_intent

    return INTENT_OUT_OF_SCOPE


def is_audio_silent(
    audio_path: str | Path,
    threshold_db: float = AUDIO_SILENCE_THRESHOLD_DB,
    min_active_ratio: float = AUDIO_MIN_ACTIVE_RATIO,
) -> bool:
    """Retourne True si l'audio est vide ou trop silencieux."""
    try:
        waveform, sr = _load_audio_for_energy(audio_path)
    except Exception:
        return True

    if waveform.numel() == 0:
        return True

    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        import torchaudio

        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    waveform = waveform.float().squeeze()
    if waveform.numel() < 512:
        return True

    frame_length = 2048
    hop_length = 512
    if waveform.numel() < frame_length:
        import torch

        rms = torch.sqrt(torch.mean(waveform**2)).reshape(1)
    else:
        import torch

        frames = waveform.unfold(0, frame_length, hop_length)
        rms = torch.sqrt(torch.mean(frames**2, dim=1))

    import torch

    db = 20 * torch.log10(rms + 1e-10)
    active_ratio = float((db > threshold_db).sum().item() / max(len(db), 1))
    return active_ratio < min_active_ratio


def detect_question_intent(question: str) -> str:
    """Ancienne granularité utilisée pour reformuler les chunks RAG."""
    q = _normalize(question)

    if _contains_any(
        q,
        [
            "problem",
            "probleme",
            "problème",
            "symptom",
            "symptôme",
            "leaves",
            "leaf",
            "feuilles",
            "reddish",
            "spots",
            "taches",
        ],
    ):
        return "diagnosis"
    if _contains_any(q, ["qu'est-ce", "qu est ce", "what is", "what's", "menene"]):
        return "definition"
    if _contains_any(q, ["pluie", "pleut", "rain", "rainy", "damina", "ndungu"]):
        return "rain"
    if _contains_any(q, ["entretenir", "entretien", "care", "watering", "arroser"]):
        return "care"
    if _contains_any(q, ["traiter", "treat", "soigner", "magani"]):
        return "treatment"
    return "general"


def format_retrieval_answer(question: str, chunk: dict, target_lang: str) -> str:
    intent = detect_question_intent(question)
    title = chunk["title"]
    body = _strip_title(chunk["text"], title)
    disease_name = _display_title(title)

    if intent == "definition":
        return _definition_answer(disease_name, body, target_lang)
    if intent == "rain":
        return _rain_answer(body, target_lang)
    if intent == "care":
        return _care_answer(body, target_lang)
    if intent == "diagnosis":
        return _diagnosis_answer(disease_name, body, target_lang)
    if intent == "treatment":
        return _treatment_answer(disease_name, body, target_lang)
    return _general_answer(body, target_lang)


def _classify_social_intent(text: str, lang: str) -> str | None:
    for intent, lexicon in [
        (INTENT_GREETING, GREETINGS),
        (INTENT_IDENTITY, IDENTITY_KEYWORDS),
        (INTENT_THANKS, THANKS_KEYWORDS),
        (INTENT_GOODBYE, GOODBYE_KEYWORDS),
    ]:
        candidates = lexicon.get(lang, []) + lexicon.get("fr", []) + lexicon.get("en", [])
        if _matches_short_social_text(text, candidates):
            return intent
    return None


def _matches_short_social_text(text: str, candidates: list[str]) -> bool:
    if len(text.split()) > 8:
        return False
    return any(term in text for term in candidates)


def _has_disease_retrieval_match(text: str, lang: str) -> bool:
    try:
        from .retrieval import retrieve

        results = retrieve(text, lang=lang, k=1)
    except Exception:
        return False

    return bool(results and results[0][0] >= DISEASE_RETRIEVAL_THRESHOLD)


def _classify_semantic_intent(text: str) -> str | None:
    try:
        import numpy as np

        from .retrieval import encoder
    except Exception:
        return None

    labels = []
    prototypes = []
    for intent, samples in PROTOTYPES.items():
        labels.extend([intent] * len(samples))
        prototypes.extend(samples)

    query_emb = encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    proto_emb = encoder.encode(prototypes, normalize_embeddings=True, convert_to_numpy=True)
    sims = (proto_emb @ query_emb.T).flatten()
    best_idx = int(np.argmax(sims))
    if float(sims[best_idx]) >= SEMANTIC_THRESHOLD:
        return labels[best_idx]
    return None


def _is_unclear_text(text: str) -> bool:
    if len(text.strip()) < UNCLEAR_MIN_CHARS:
        return True

    words = [word for word in re.findall(r"[\wɗɓƙƴ'’-]+", text.lower()) if word]
    if not words:
        return True
    if len(words) == 1 and len(words[0]) > 12 and len(VOWELS_RE.findall(words[0])) <= 2:
        return True
    if len(words) <= 2 and all(UNCLEAR_WORD_RE.match(word) for word in words):
        known_fragments = TOMATO_TERMS + DISEASE_TERMS + GENERIC_TOMATO_TERMS
        if not _contains_any(text, known_fragments):
            return True
    return False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _load_audio_for_energy(audio_path: str | Path):
    import torchaudio

    try:
        return torchaudio.load(str(audio_path))
    except Exception as exc:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return torchaudio.load(str(wav_path))
        finally:
            wav_path.unlink(missing_ok=True)


def _strip_title(text: str, title: str) -> str:
    answer = text
    if answer.startswith(title):
        answer = answer[len(title) :].lstrip(". ").strip()
    return answer


def _display_title(title: str) -> str:
    return title.split("(")[0].split("—")[0].strip()


def _definition_answer(disease_name: str, body: str, lang: str) -> str:
    first_facts = _neutralize_diagnostic_phrasing(_first_sentences(body, 3), lang)
    templates = {
        "fr": (
            f"{disease_name} est un problème qui peut toucher les tomates. "
            f"Voici ce qu'il faut retenir : {first_facts} "
            "Cette réponse est informative : elle ne veut pas dire que votre plante est malade."
        ),
        "en": (
            f"{disease_name} is a problem that can affect tomatoes. "
            f"Key points: {first_facts} "
            "This is informational and does not mean your plant is diseased."
        ),
        "ha": f"{disease_name} cuta ce da ka iya kama tumatur. Abubuwan da suka fi muhimmanci: {first_facts}",
        "ff": f"{disease_name} ko caɗeele nde waawi heɓde tomati. Ɓuri teeŋtude: {first_facts}",
    }
    return templates.get(lang, templates["fr"])


def _rain_answer(body: str, lang: str) -> str:
    tips = _sentences_with_keywords(
        body,
        ["pluie", "rain", "humidité", "humidity", "eau", "water", "mouill", "wet"],
        fallback_count=4,
    )
    templates = {
        "fr": (
            "La pluie augmente l'humidité autour des feuilles de tomate, donc elle peut favoriser certaines maladies. "
            f"Conseils pratiques : {tips} "
            "Cela ne signifie pas que votre plante est déjà malade ; surveillez les feuilles après les pluies."
        ),
        "en": (
            "Rain increases humidity around tomato leaves, so it can favor some diseases. "
            f"Practical advice: {tips} "
            "This does not mean your plant is already diseased; check the leaves after rain."
        ),
        "ha": f"Ruwa ko damina na kara danshi a ganyen tumatur, kuma hakan na iya taimaka wa wasu cututtuka. Shawara: {tips}",
        "ff": f"Ndiyam toɓɓere maa ndungu ɓeydata heccere e haako tomati, ɗum waawi wallude caɗeele. Ɗowgol: {tips}",
    }
    return templates.get(lang, templates["fr"])


def _care_answer(body: str, lang: str) -> str:
    tips = _neutralize_diagnostic_phrasing(_first_sentences(body, 5), lang)
    templates = {
        "fr": f"Pour bien entretenir vos tomates, concentrez-vous sur la prévention et l'observation régulière. {tips}",
        "en": f"To care for tomato plants well, focus on prevention and regular observation. {tips}",
        "ha": f"Don kula da tumatur da kyau, ka fi mayar da hankali kan kariya da dubawa akai-akai. {tips}",
        "ff": f"Ngam toppitorde tomati no moƴƴi, waɗtu hakkille e haɗde caɗeele e yiytude ɗe sahaa kala. {tips}",
    }
    return templates.get(lang, templates["fr"])


def _diagnosis_answer(disease_name: str, body: str, lang: str) -> str:
    tips = _first_sentences(body, 5)
    templates = {
        "fr": (
            f"D'après les mots de votre question, cela peut faire penser à {disease_name}, "
            f"mais une photo claire des feuilles est nécessaire pour confirmer. À vérifier : {tips}"
        ),
        "en": (
            f"Based on your words, this may look like {disease_name}, "
            f"but a clear photo of the leaves is needed to confirm. Check this: {tips}"
        ),
        "ha": f"Daga bayanin tambayarka, yana iya kama da {disease_name}, amma ana bukatar hoto bayyananne domin tabbatarwa. {tips}",
        "ff": f"E dow ko naamnaade maa holli, ɗum waawi nanndude e {disease_name}, kono natal laaɓɗo haako ina walla ngam teeŋtinde. {tips}",
    }
    return templates.get(lang, templates["fr"])


def _treatment_answer(disease_name: str, body: str, lang: str) -> str:
    tips = _first_sentences(body, 5)
    templates = {
        "fr": f"Si {disease_name} est confirmé sur vos tomates, voici les actions utiles : {tips}",
        "en": f"If {disease_name} is confirmed on your tomatoes, useful actions are: {tips}",
        "ha": f"Idan an tabbatar da {disease_name} a tumatur, ga abin da za a yi: {tips}",
        "ff": f"So {disease_name} teeŋtinaama e tomati maa, ɗum ko ko waɗeten: {tips}",
    }
    return templates.get(lang, templates["fr"])


def _general_answer(body: str, lang: str) -> str:
    tips = _neutralize_diagnostic_phrasing(_first_sentences(body, 4), lang)
    templates = {
        "fr": f"Voici une réponse utile pour la culture de tomate : {tips}",
        "en": f"Here is useful guidance for tomato growing: {tips}",
        "ha": f"Ga shawara mai amfani ga noman tumatur: {tips}",
        "ff": f"Ɗum ko ɗowgol nafowol e remde tomati: {tips}",
    }
    return templates.get(lang, templates["fr"])


def _first_sentences(text: str, count: int) -> str:
    sentences = _split_sentences(text)
    return " ".join(sentences[:count]).strip()


def _sentences_with_keywords(text: str, keywords: list[str], fallback_count: int) -> str:
    sentences = _split_sentences(text)
    selected = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in keywords)
    ]
    if not selected:
        selected = sentences[:fallback_count]
    return " ".join(selected[:fallback_count]).strip()


def _split_sentences(text: str) -> list[str]:
    return [part.strip() + "." for part in re.split(r"\.\s+", text) if part.strip()]


def _neutralize_diagnostic_phrasing(text: str, lang: str) -> str:
    replacements = {
        "fr": {
            "Votre tomate a": "Les tomates atteintes peuvent avoir",
            "Vos feuilles ont": "Les feuilles atteintes peuvent avoir",
            "Vos feuilles jaunissent": "Les feuilles atteintes peuvent jaunir",
            "Votre tomate est en bonne santé": "Une tomate en bonne santé",
        },
        "en": {
            "Your tomato has": "Affected tomato plants may have",
            "Your leaves have": "Affected leaves may have",
            "Your leaves are": "Affected leaves may be",
            "Your tomato is in good health": "A healthy tomato plant is in good health",
        },
    }
    for old, new in replacements.get(lang, {}).items():
        text = text.replace(old, new)
    return text
