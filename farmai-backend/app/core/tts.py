from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import torch
from transformers import AutoTokenizer as VitsTokenizer
from transformers import VitsModel

from ..config import AUDIO_OUTPUT_DIR, DEVICE, TTS_LANG_CODES

_tts_cache: dict[str, tuple[VitsModel, VitsTokenizer]] = {}


def get_tts(lang: str):
    if lang in _tts_cache:
        return _tts_cache[lang]

    iso_code = TTS_LANG_CODES.get(lang)
    if not iso_code:
        raise ValueError(f"Langue non supportée pour le TTS : {lang}")

    model_id = f"facebook/mms-tts-{iso_code}"
    print(f"Chargement TTS {lang} ({model_id})")
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    tokenizer = VitsTokenizer.from_pretrained(model_id)

    _tts_cache[lang] = (model, tokenizer)
    return model, tokenizer


def synthesize_audio(text: str, lang: str = "ha", output_path: str | Path | None = None) -> str:
    model, tokenizer = get_tts(lang)

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.cpu().numpy().squeeze()
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)

    if output_path is None:
        output_path = AUDIO_OUTPUT_DIR / f"farmai_{lang}_{abs(hash(text)) % 100000}.wav"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), rate=model.config.sampling_rate, data=audio_int16)
    return str(output_path)
