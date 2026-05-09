from app.pipeline import farmai_assistant


def test_text_pipeline_with_mocks(monkeypatch):
    def fake_generate_response_hybrid(question, target_lang="fr", k=2):
        return {
            "question": question,
            "target_lang": target_lang,
            "answer": "Conseil test",
            "sources": [{"score": 0.9, "title": "Source test", "lang": "Français"}],
        }

    monkeypatch.setattr("app.pipeline.generate_response_hybrid", fake_generate_response_hybrid)

    result = farmai_assistant(
        text_question="Comment traiter le mildiou ?",
        target_lang="fr",
        return_audio=False,
    )

    assert result["question"] == "Comment traiter le mildiou ?"
    assert result["answer_text"] == "Conseil test"
    assert result["sources"][0]["title"] == "Source test"
