from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from ..config import (
    CONFIDENCE_THRESHOLD,
    DISEASE_CLASSES,
    ENTROPY_THRESHOLD,
    TFLITE_MODEL_PATH,
)

if not TFLITE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle TFLite introuvable : {TFLITE_MODEL_PATH}")

print(f"Chargement du modèle TFLite : {TFLITE_MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modele TFLite pret")


def normalized_entropy(probs: np.ndarray | list[float]) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(len(probs)))


def get_image_reliability_metrics(probs: np.ndarray | list[float]) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float32)
    return {
        "max_confidence": float(np.max(probs)),
        "entropy": normalized_entropy(probs),
    }


def is_valid_leaf_image(probs: np.ndarray | list[float]) -> bool:
    metrics = get_image_reliability_metrics(probs)
    return (
        metrics["max_confidence"] >= CONFIDENCE_THRESHOLD
        and metrics["entropy"] <= ENTROPY_THRESHOLD
    )


def detect_disease_from_image(
    image_path: str | Path,
    debug: bool = False,
) -> tuple[str, float, list[float]]:
    """
    Charge une image, la prétraite, et retourne
    (classe_prédite, confiance, probabilités complètes).
    Gère les modèles TFLite float et quantifiés.
    """
    img = Image.open(image_path).convert("RGB")
    _, h, w, _ = input_details[0]["shape"]
    img = img.resize((w, h))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    if input_details[0]["dtype"] == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        if scale:
            arr = (arr / scale + zero_point).astype(np.uint8)
        else:
            arr = (arr * 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    if output_details[0]["dtype"] in (np.uint8, np.int8):
        scale, zero_point = output_details[0]["quantization"]
        if scale:
            output = (output.astype(np.float32) - zero_point) * scale

    output = output.astype(np.float32)
    if output.max() > 1.0 or output.min() < 0.0:
        exp_output = np.exp(output - output.max())
        probs = exp_output / exp_output.sum()
    else:
        probs = output

    if debug:
        for cls, proba in sorted(zip(DISEASE_CLASSES, probs), key=lambda x: -x[1]):
            print(f"{proba:.4f} {cls}")

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return DISEASE_CLASSES[pred_idx], confidence, probs.astype(float).tolist()
