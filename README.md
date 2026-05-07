# FarmAI Backend

Backend FastAPI pour le pipeline FarmAI Cameroun : vision TFLite, RAG FAISS, génération Qwen, TTS MMS et ASR MMS optionnel.

## Structure

```text
farmai-backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── pipeline.py
│   ├── schemas.py
│   └── core/
├── models/
│   ├── farmai_disease_detector.tflite
│   └── faiss_index/
├── data/
├── tests/
├── requirements.txt
├── railway.json
├── render.yaml
└── Dockerfile
```

## Lancement local

```bash
cd farmai-backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API disponible sur `http://127.0.0.1:8000`.

## Documentation Swagger

FastAPI génère automatiquement la documentation interactive :

- Swagger UI : `http://127.0.0.1:8000/docs`
- ReDoc : `http://127.0.0.1:8000/redoc`
- Schéma OpenAPI JSON : `http://127.0.0.1:8000/openapi.json`

En production, remplace simplement `http://127.0.0.1:8000` par l'URL Railway ou Render.

## Endpoints

- `GET /`
- `GET /health`
- `POST /ask/text`
- `POST /ask/image`
- `POST /ask/audio`

Exemple texte :

```bash
curl -X POST http://127.0.0.1:8000/ask/text \
  -H "Content-Type: application/json" \
  -d '{"text_question":"Comment traiter le mildiou tardif ?","target_lang":"fr","return_audio":false,"k":2}'
```

Les réponses audio générées sont servies sous `/audio/<fichier.wav>`.

## Langues supportées

Le champ `target_lang` accepte :

- `fr` : français
- `en` : anglais
- `ha` : Hausa
- `ff` : Fulfulde/Fulfulbe

## Déploiement Railway

Le projet est prêt pour Railway avec `railway.json`, `Dockerfile` et `Procfile`.

Depuis Railway :

1. Crée un nouveau projet depuis le repository GitHub.
2. Choisis `farmai-backend` comme répertoire du service si Railway te le demande.
3. Ajoute les variables d'environnement :

```bash
DEVICE=cpu
CORS_ORIGINS=*
```

4. Railway expose automatiquement la variable `$PORT`; le Dockerfile et le `Procfile` l'utilisent déjà.
5. Healthcheck : `/health`.

Après déploiement :

- Swagger : `https://<ton-service>.up.railway.app/docs`
- OpenAPI : `https://<ton-service>.up.railway.app/openapi.json`
- Healthcheck : `https://<ton-service>.up.railway.app/health`

## Déploiement Render

Un fichier `render.yaml` est aussi inclus.

Sur Render :

1. Crée un Web Service depuis le repository.
2. Utilise Docker comme runtime.
3. Si le repository contient plusieurs dossiers, indique `farmai-backend` comme root directory.
4. Healthcheck : `/health`.
5. Variables recommandées :

```bash
DEVICE=cpu
CORS_ORIGINS=*
```

## CORS frontend

Par défaut, `CORS_ORIGINS=*` permet à l'équipe frontend d'appeler l'API depuis n'importe quelle origine.

Pour la production finale, remplace `*` par les domaines frontend séparés par des virgules :

```bash
CORS_ORIGINS=https://frontend.example.com,https://staging.example.com
```

## Notes de capacité

Le backend télécharge et charge des modèles HuggingFace lourds au démarrage ou au premier appel selon le module :

- Qwen pour `fr` et `en`
- SentenceTransformer pour le RAG
- MMS-TTS au premier appel audio
- MMS-ASR au premier appel audio entrant

Prévois une instance avec assez de RAM et un temps de démarrage suffisamment long. Pour un MVP sans audio, envoie `return_audio=false` depuis le frontend.
