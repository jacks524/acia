import pytest

from app.pipeline import ImageNotRecognizedError, farmai_assistant


def _patch_common(monkeypatch):
    def fake_generate_response_hybrid(question, target_lang="fr", k=2):
        return {
            "question": question,
            "target_lang": target_lang,
            "answer": "Conseil test",
            "sources": [{"score": 0.9, "title": "Source test", "lang": "Français"}],
        }

    monkeypatch.setattr("app.pipeline.generate_response_hybrid", fake_generate_response_hybrid)
    monkeypatch.setattr("app.pipeline.synthesize_audio", lambda text, lang="fr": None)


def test_tomato_leaf_passes_validation(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    image_path = tmp_path / "leaf.jpg"
    image_path.write_bytes(b"fake image")

    probs = [0.92, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01]
    monkeypatch.setattr(
        "app.pipeline.detect_disease_from_image",
        lambda path: ("Tomato___Late_blight", 0.92, probs),
    )
    monkeypatch.setattr(
        "app.pipeline.is_plant_leaf",
        lambda path: pytest.fail("CLIP backup should not run for a confident leaf"),
    )

    result = farmai_assistant(
        image_path=str(image_path),
        target_lang="fr",
        return_audio=False,
    )

    assert result["detected_disease"] == "Tomato___Late_blight"
    assert result["answer_text"] == "Conseil test"


def test_unrelated_image_rejected_after_level_1_and_clip(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    image_path = tmp_path / "person.jpg"
    image_path.write_bytes(b"fake image")

    probs = [0.1] * 10
    monkeypatch.setattr(
        "app.pipeline.detect_disease_from_image",
        lambda path: ("Tomato___healthy", 0.1, probs),
    )
    monkeypatch.setattr("app.pipeline.is_plant_leaf", lambda path: False)

    with pytest.raises(ImageNotRecognizedError) as exc:
        farmai_assistant(
            image_path=str(image_path),
            target_lang="fr",
            return_audio=False,
        )

    assert exc.value.payload["error"] == "image_not_recognized"
    assert exc.value.payload["error_code"] == 422
    assert exc.value.payload["max_confidence"] == pytest.approx(0.1)
    assert exc.value.payload["entropy"] > 0.9


def test_green_object_rejected_by_clip_backup(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    image_path = tmp_path / "green_object.jpg"
    image_path.write_bytes(b"fake image")

    probs = [0.6, 0.09, 0.08, 0.07, 0.06, 0.04, 0.02, 0.02, 0.01, 0.01]
    monkeypatch.setattr(
        "app.pipeline.detect_disease_from_image",
        lambda path: ("Tomato___Bacterial_spot", 0.6, probs),
    )
    monkeypatch.setattr("app.pipeline.is_plant_leaf", lambda path: False)

    with pytest.raises(ImageNotRecognizedError) as exc:
        farmai_assistant(
            image_path=str(image_path),
            target_lang="fr",
            return_audio=False,
        )

    assert exc.value.payload["error"] == "image_not_recognized"
    assert exc.value.payload["max_confidence"] == pytest.approx(0.6)
