# FarmAI Backend

Backend FastAPI pour FarmAI Cameroun : vision TFLite, recherche FAISS, réponses quadrilingues et audio optionnel.

## Documentation API

FastAPI génère automatiquement la documentation :

- Swagger UI : `/docs`
- ReDoc : `/redoc`
- OpenAPI JSON : `/openapi.json`
- Healthcheck : `/health`

En production :

```text
https://<ton-service>/docs
```

## Langues

`target_lang` accepte :

- `fr` : français
- `en` : anglais
- `ha` : Hausa
- `ff` : Fulfulde/Fulfulbe

## Variables d'environnement Railway/Render CPU

Sur Railway ou Render CPU, garde le LLM désactivé pour éviter le crash mémoire :

```env
DEVICE=cpu
USE_LLM=false
ENCODER_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_NAME=Qwen/Qwen2.5-3B-Instruct
ASR_MODEL_NAME=facebook/mms-1b-all
CORS_ORIGINS=*
```

Avec `USE_LLM=false`, le backend répond en retrieval-only pour les 4 langues et ne charge pas Qwen.

## Variables d'environnement GPU

Sur un environnement avec GPU CUDA et assez de RAM/VRAM :

```env
DEVICE=cuda
USE_LLM=true
ENCODER_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_NAME=Qwen/Qwen2.5-3B-Instruct
ASR_MODEL_NAME=facebook/mms-1b-all
CORS_ORIGINS=*
```

## Lancement local

```bash
cd farmai-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API locale :

```text
http://127.0.0.1:8000/docs
```

## Exemple

```bash
curl -X POST http://127.0.0.1:8000/ask/text \
  -H "Content-Type: application/json" \
  -d '{"text_question":"Comment traiter le mildiou tardif ?","target_lang":"fr","return_audio":false,"k":2}'
```

## Note mémoire

Le log `Killed` pendant `Loading checkpoint shards` indique un manque de RAM au chargement de Qwen. Sur CPU gratuit ou petit plan Railway/Render, utilise `USE_LLM=false`.

Pour un MVP stable, le frontend devrait aussi envoyer `return_audio=false`, car MMS-TTS et MMS-ASR sont lourds.
