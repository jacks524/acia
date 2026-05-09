from app.core.intent import detect_question_intent, format_retrieval_answer


FR_LATE_BLIGHT = {
    "title": "Mildiou tardif (Late Blight)",
    "lang": "fr",
    "lang_label": "Français",
    "text": (
        "Mildiou tardif (Late Blight). Votre tomate a des taches brunes qui "
        "s'étendent vite, c'est le mildiou tardif. C'est une maladie très "
        "dangereuse. Cette maladie aime la pluie et le froid. Arrosez toujours "
        "au pied, jamais sur les feuilles."
    ),
}


def test_definition_question_does_not_diagnose_plant():
    answer = format_retrieval_answer(
        "Qu'est-ce que le mildiou ?",
        FR_LATE_BLIGHT,
        "fr",
    )

    assert detect_question_intent("Qu'est-ce que le mildiou ?") == "definition"
    assert "ne veut pas dire que votre plante est malade" in answer


def test_rain_question_returns_prevention_not_diagnosis():
    answer = format_retrieval_answer(
        "Que faire quand il pleut beaucoup ?",
        FR_LATE_BLIGHT,
        "fr",
    )

    assert detect_question_intent("Que faire quand il pleut beaucoup ?") == "rain"
    assert answer.startswith("La pluie augmente")
    assert "déjà malade" in answer


def test_symptom_question_is_prudent_diagnosis():
    answer = format_retrieval_answer(
        "My tomato leaves are looking reddish, what is the problem?",
        FR_LATE_BLIGHT,
        "en",
    )

    assert detect_question_intent("leaves are reddish") == "diagnosis"
    assert "may look like" in answer
    assert "needed to confirm" in answer
