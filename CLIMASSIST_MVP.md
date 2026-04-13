# ClimAssist MVP

This branch contains a lightweight deployment path for a shareable farming-advisory app.

What changed from the full ClimSight stack:

- no local climate-data bundles required
- no Chroma or RAG required
- no sandboxed Python analysis
- no WebSocket/session dependency for the main flow
- works with free Open-Meteo forecast data
- upgrades automatically to OpenRouter or OpenAI if a model key is configured

## What It Does

The app lets a user:

- choose a location on a map
- select crop and farming stage
- ask a practical farming question
- receive a grounded advisory based on the next 7 days of forecast data

If `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is set, the answer is synthesized by an LLM.
If not, the app still works in deterministic fallback mode.

## Fast Local Run

### 1. Python dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Optional environment

Copy `.env.example` to `.env` and set one provider if you want richer AI output:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=google/gemma-3-4b-it
```

If no key is set, the app still runs.

### 4. Development mode

Backend:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Single-Service Deploy

If you want one deployable service instead of separate frontend and backend:

```bash
cd frontend
npm run build
cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

After `frontend/dist` exists, FastAPI serves the built frontend at `/`.

## Suggested Minimal Hosting

For the least infrastructure:

- deploy one FastAPI service
- build the frontend during deploy
- set `OPENROUTER_API_KEY` only if you want AI-generated Hausa or richer narrative responses

This keeps the runtime footprint low and avoids shipping large climate files or vector databases.
