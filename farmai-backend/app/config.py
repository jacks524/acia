from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

TFLITE_MODEL_PATH = MODELS_DIR / "farmai_disease_detector.tflite"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index" / "farmai.index"
CHUNKS_META_PATH = MODELS_DIR / "faiss_index" / "chunks_meta.json"
LABELS_PATH = DATA_DIR / "farmai_labels.json"

ENCODER_NAME = os.getenv(
    "ENCODER_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LLM_NAME = os.getenv("LLM_NAME", "Qwen/Qwen2.5-3B-Instruct")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "facebook/mms-1b-all")

DEVICE = os.getenv("DEVICE", "cpu")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

TTS_LANG_CODES = {
    "ha": "hau",
    "ff": "ful",
    "fr": "fra",
    "en": "eng",
}

ASR_LANG_CODES = {
    "ha": "hau",
    "ff": "ful",
    "fr": "fra",
    "en": "eng",
}

LANG_LABELS = {
    "ha": "Hausa",
    "ff": "Fulfulde",
    "fr": "Français",
    "en": "English",
}

_FALLBACK_DISEASE_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

_LABEL_TO_CLASS = {
    "Bacterial_spot": "Tomato___Bacterial_spot",
    "Early_blight": "Tomato___Early_blight",
    "healthy": "Tomato___healthy",
    "Late_blight": "Tomato___Late_blight",
    "Leaf_Mold": "Tomato___Leaf_Mold",
    "Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Spider_mites": "Tomato___Spider_mites",
    "Target_Spot": "Tomato___Target_Spot",
    "Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "YellowLeaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}


def _load_disease_classes() -> list[str]:
    if not LABELS_PATH.exists():
        return _FALLBACK_DISEASE_CLASSES

    with LABELS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    labels = payload.get("class_names", [])
    classes = [_LABEL_TO_CLASS.get(label, label) for label in labels]
    return classes or _FALLBACK_DISEASE_CLASSES


DISEASE_CLASSES = _load_disease_classes()

DISEASE_KEYWORDS = {
    "Tomato___Bacterial_spot": "Bacterial Spot tache bactérienne",
    "Tomato___Early_blight": "Early Blight alternariose",
    "Tomato___healthy": "Healthy Plant plante saine",
    "Tomato___Late_blight": "Late Blight mildiou tardif",
    "Tomato___Leaf_Mold": "Leaf Mold moisissure feuilles",
    "Tomato___Septoria_leaf_spot": "Septoria septoriose tache",
    "Tomato___Spider_mites": "Spider Mites acariens insectes",
    "Tomato___Target_Spot": "Target Spot taches circulaires",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus virus mosaïque",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Yellow Leaf Curl Virus enroulement jaune"
    ),
}

AUDIO_OUTPUT_DIR = BASE_DIR / "outputs" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
