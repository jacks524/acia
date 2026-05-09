import re


DEFINITION_TERMS = [
    "qu'est ce",
    "qu'est-ce",
    "c'est quoi",
    "definition",
    "définition",
    "what is",
    "what's",
    "meaning",
    "menene",
    "ko woni",
    "hol ko",
]

CARE_TERMS = [
    "entretenir",
    "entretien",
    "prendre soin",
    "mieux entretenir",
    "arroser",
    "arrosage",
    "compost",
    "engrais",
    "mauvaises herbes",
    "care",
    "maintain",
    "maintenance",
    "water",
    "watering",
    "fertilizer",
    "weeds",
]

RAIN_TERMS = [
    "pluie",
    "saison des pluies",
    "pleut",
    "rain",
    "rainy",
    "heavy rain",
    "damina",
    "ndungu",
]

TREATMENT_TERMS = [
    "traiter",
    "soigner",
    "guérir",
    "guerir",
    "appliquer",
    "pulvériser",
    "pulveriser",
    "fongicide",
    "treat",
    "cure",
    "apply",
    "spray",
    "magani",
]

DIAGNOSIS_TERMS = [
    "probleme",
    "problème",
    "problem",
    "symptome",
    "symptôme",
    "symptom",
    "tache",
    "taches",
    "spot",
    "spots",
    "reddish",
    "red",
    "yellow",
    "jaune",
    "brun",
    "brown",
    "curl",
    "enroul",
    "feuilles",
    "leaves",
    "leaf",
]


def detect_question_intent(question: str) -> str:
    q = _normalize(question)

    if _contains_any(q, DIAGNOSIS_TERMS):
        return "diagnosis"
    if _contains_any(q, DEFINITION_TERMS):
        return "definition"
    if _contains_any(q, RAIN_TERMS):
        return "rain"
    if _contains_any(q, CARE_TERMS):
        return "care"
    if _contains_any(q, TREATMENT_TERMS):
        return "treatment"
    return "general"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


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
        "ha": (
            f"{disease_name} cuta ce da ka iya kama tumatur. "
            f"Abubuwan da suka fi muhimmanci: {first_facts}"
        ),
        "ff": (
            f"{disease_name} ko caɗeele nde waawi heɓde tomati. "
            f"Ɓuri teeŋtude: {first_facts}"
        ),
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
        "ha": (
            "Ruwa ko damina na kara danshi a ganyen tumatur, kuma hakan na iya taimaka wa wasu cututtuka. "
            f"Shawara: {tips}"
        ),
        "ff": (
            "Ndiyam toɓɓere maa ndungu ɓeydata heccere e haako tomati, ɗum waawi wallude caɗeele. "
            f"Ɗowgol: {tips}"
        ),
    }
    return templates.get(lang, templates["fr"])


def _care_answer(body: str, lang: str) -> str:
    tips = _neutralize_diagnostic_phrasing(_first_sentences(body, 5), lang)
    templates = {
        "fr": (
            "Pour bien entretenir vos tomates, concentrez-vous sur la prévention et l'observation régulière. "
            f"{tips}"
        ),
        "en": (
            "To care for tomato plants well, focus on prevention and regular observation. "
            f"{tips}"
        ),
        "ha": f"Don kula da tumatur da kyau, ka fi mayar da hankali kan kariya da dubawa akai-akai. {tips}",
        "ff": f"Ngam toppitorde tomati no moƴƴi, waɗtu hakkille e haɗde caɗeele e yiytude ɗe sahaa kala. {tips}",
    }
    return templates.get(lang, templates["fr"])


def _diagnosis_answer(disease_name: str, body: str, lang: str) -> str:
    tips = _first_sentences(body, 5)
    templates = {
        "fr": (
            f"D'après les mots de votre question, cela peut faire penser à {disease_name}, "
            "mais une photo claire des feuilles est nécessaire pour confirmer. "
            f"À vérifier : {tips}"
        ),
        "en": (
            f"Based on your words, this may look like {disease_name}, "
            "but a clear photo of the leaves is needed to confirm. "
            f"Check this: {tips}"
        ),
        "ha": (
            f"Daga bayanin tambayarka, yana iya kama da {disease_name}, "
            f"amma ana bukatar hoto bayyananne domin tabbatarwa. {tips}"
        ),
        "ff": (
            f"E dow ko naamnaade maa holli, ɗum waawi nanndude e {disease_name}, "
            f"kono natal laaɓɗo haako ina walla ngam teeŋtinde. {tips}"
        ),
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
