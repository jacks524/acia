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
USE_CLIP_VALIDATOR=true
CLIP_ALWAYS_VALIDATE=true
CONFIDENCE_THRESHOLD=0.65
ENTROPY_THRESHOLD=0.55
ENCODER_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_NAME=Qwen/Qwen2.5-3B-Instruct
ASR_MODEL_NAME=facebook/mms-1b-all
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
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

## Validation des images

L'endpoint `/ask/image` rejette les images qui ne ressemblent pas à une feuille de tomate.

Niveau 1 : le modèle TFLite retourne les 10 probabilités softmax. Le backend calcule :

- `max_confidence` : probabilité de la classe la plus probable
- `entropy` : entropie de Shannon normalisée entre 0 et 1

Une image est acceptée au niveau 1 seulement si :

```env
CONFIDENCE_THRESHOLD=0.65
ENTROPY_THRESHOLD=0.55
```

Impact du calibrage :

- seuil de confiance trop bas : laisse passer plus d'images hors-sujet
- seuil de confiance trop haut : rejette plus de vraies feuilles
- seuil d'entropie trop haut : accepte des prédictions trop incertaines
- seuil d'entropie trop bas : rejette des feuilles valides si le modèle hésite

Niveau 2 : le backend utilise CLIP (`openai/clip-vit-base-patch32`) pour vérifier si l'image ressemble à une feuille/plante. Par défaut, `CLIP_ALWAYS_VALIDATE=true` force cette vérification même quand le TFLite est confiant, parce qu'un classifieur fermé peut donner une forte confiance sur une image hors-sujet. Si CLIP répond non, l'API retourne HTTP `422` avec :

```json
{
  "error": "image_not_recognized",
  "error_code": 422,
  "message_fr": "L'image ne semble pas être une feuille de tomate. Veuillez prendre une photo claire d'une feuille.",
  "message_en": "The image does not appear to be a tomato leaf. Please take a clear photo of a leaf.",
  "message_ha": "Hoton ba kamar ganyen tumatur ba ne. Da fatan za ka dauki hoto bayyananne na ganye.",
  "message_ff": "Natal ngal nanndaani e haako tomati. Yiɗde ƴetto natal laaɓɗo no haako waɗi.",
  "max_confidence": 0.42,
  "entropy": 0.81
}
```

CLIP peut être désactivé avec :

```env
USE_CLIP_VALIDATOR=false
```

Pour utiliser CLIP seulement en backup quand le niveau 1 échoue :

```env
CLIP_ALWAYS_VALIDATE=false
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

## Appel audio depuis le frontend

L'endpoint attend un multipart avec le champ exact `audio` :

```bash
curl -X POST "https://acia-production.up.railway.app/ask/audio" \
  -F "audio=@audio.m4a" \
  -F "target_lang=fr" \
  -F "return_audio=true" \
  -F "k=2"
```

Dans une app mobile, ne force pas manuellement le header `Content-Type: multipart/form-data` si tu utilises `FormData`. Laisse Axios/fetch ajouter le `boundary`, sinon le serveur peut recevoir un multipart invalide.

Le container installe `ffmpeg` pour lire les fichiers `m4a`, `mp3`, `opus` et `wav`.
