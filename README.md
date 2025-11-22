## Fish Market Voice Bot

Async FastAPI service that turns customer voice queries into Malayalam or English answers powered by [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition, OpenAI responses for reasoning, and OpenAI TTS for speech synthesis. Fish rates are vectorized from `fish_rate.json` so the bot can answer inventory/price questions quickly without a database.

### Features
- Voice → text → response → voice loop handled end-to-end in `FishMarketVoiceBot`.
- Malayalam detection: Malayalam queries receive Malayalam replies; others default to English.
- Fish catalog semantic retrieval via OpenAI embeddings for accurate price/stock lookup.
- Async FastAPI endpoint `/voice-chat` returning MP3 audio plus transcript/reply metadata headers.
- Dockerfile + docker-compose for easy deployment with GPU-less Whisper (defaults to `small`).

### Prerequisites
- Python 3.11+
- FFmpeg (installed automatically inside the container)
- OpenAI API key stored as `OPENAI_API_KEY`

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env  # fill in your key/models
uvicorn app.main:app --reload
```

Send a request:
```bash
curl -X POST http://localhost:8000/voice-chat \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@sample.wav" \
  --output reply.mp3
```

### Docker
```bash
docker compose up --build
```

### Configuration
All tunable settings live in `.env` (mirrors `Settings` in `app/config.py`). Adjust model names, voices, or fish JSON path there.

