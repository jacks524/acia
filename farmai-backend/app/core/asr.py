import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

from ..config import ASR_LANG_CODES, ASR_MODEL_NAME, DEVICE

_asr_processor = None
_asr_model = None


def _get_asr():
    global _asr_processor, _asr_model
    if _asr_processor is None or _asr_model is None:
        print(f"Chargement ASR : {ASR_MODEL_NAME}")
        _asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)
        _asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
        _asr_model.eval()
    return _asr_processor, _asr_model


def transcribe_audio(audio_path: str, lang: str = "ha") -> str:
    processor, model = _get_asr()
    mms_lang = ASR_LANG_CODES.get(lang, lang)

    processor.tokenizer.set_target_lang(mms_lang)
    model.load_adapter(mms_lang)

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]
