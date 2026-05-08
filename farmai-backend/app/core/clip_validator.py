from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ..config import CLIP_MODEL_NAME, DEVICE, USE_CLIP_VALIDATOR

PLANT_PROMPTS = [
    "a photo of a green plant leaf",
    "a photo of a tomato leaf",
]
NON_PLANT_PROMPTS = [
    "a photo of a person",
    "a photo of an animal",
    "a photo of an object indoor",
    "a photo of a landscape",
]
PROMPTS = PLANT_PROMPTS + NON_PLANT_PROMPTS

_processor = None
_model = None


def preload_clip_validator() -> None:
    if USE_CLIP_VALIDATOR:
        _get_clip()


def _get_clip():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    print(f"Chargement du validateur CLIP : {CLIP_MODEL_NAME}")
    _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    _model.eval()
    print(f"Validateur CLIP pret sur {DEVICE}")
    return _processor, _model


def is_plant_leaf(image_path: str | Path) -> bool:
    if not USE_CLIP_VALIDATOR:
        return False

    processor, model = _get_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    plant_probability = float(probs[: len(PLANT_PROMPTS)].sum().item())
    return plant_probability > 0.5
