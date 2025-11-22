from __future__ import annotations

import asyncio
import json
import re
import tempfile
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import boto3
import whisper

# Suppress FP16 warning on CPU (Whisper automatically falls back to FP32)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langdetect import DetectorFactory, LangDetectException, detect, detect_langs
from langcodes import Language
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.database import async_session_factory
from app.fish_vector_store import FishVectorStore, load_fish_records
from app.memory import ConversationMemory, ConversationSnippet

DetectorFactory.seed = 0

app = FastAPI(title="Fish Market Voice Bot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors for optional audio field in /chat endpoint."""
    if request.url.path == "/chat":
        errors = exc.errors()
        # Filter out audio validation errors (empty string sent as audio)
        filtered_errors = []
        for error in errors:
            error_loc = error.get("loc", [])
            # Skip audio field errors about expecting UploadFile but receiving string
            if (len(error_loc) >= 2 and error_loc[1] == "audio" and 
                "Expected UploadFile" in str(error.get("msg", "")) and
                "received: <class 'str'>" in str(error.get("msg", ""))):
                # This is the audio empty string error - ignore it
                continue
            filtered_errors.append(error)
        
        # If there are other errors, return them
        if filtered_errors:
            return JSONResponse(
                status_code=422,
                content={"detail": filtered_errors},
            )
        # If only audio error was filtered out, allow request to proceed
        # The endpoint will handle it manually
    
    # For other endpoints or if there are other errors, return normally
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )





client = AsyncOpenAI(api_key=settings.openai_api_key)
vector_store: FishVectorStore | None = None
_whisper_model: whisper.Whisper | None = None
memory_store: ConversationMemory | None = None
_supported_languages: dict[str, str] | None = None  # Map of language codes to names

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region,
)


class ChatResponse(BaseModel):
    detected_language: str = Field(..., description="Human-readable language name")
    detected_language_code: str = Field(..., min_length=2, max_length=8)
    source: Literal["text", "audio"]
    customer_text: str = Field("", description="Original customer utterance as text")
    english_text: str = Field("", description="Customer utterance converted to English")
    seller_reply_english: str = Field("", description="LLM reply in English")
    seller_reply: str = Field("", description="LLM reply translated to customer language")
    context_used: list[str] = Field(default_factory=list)
    audio_url: str | None = Field(default=None, description="S3 URL of uploaded audio file if audio was provided")


def extract_output_text(response) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if content.type == "output_text":
                chunks.append(content.text)
    return "".join(chunks).strip()


def extract_chat_completion_text(response) -> str:
    chunks: list[str] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        if not message:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(block.get("text", "") for block in content if isinstance(block, dict))
    return "".join(chunks).strip()


def load_supported_languages() -> dict[str, str]:
    """Load supported languages from JSON file."""
    global _supported_languages
    if _supported_languages is not None:
        return _supported_languages
    
    lang_path = settings.supported_languages_path
    if not lang_path.exists():
        # Default to English only if file doesn't exist
        _supported_languages = {"en": "English"}
        return _supported_languages
    
    try:
        raw_data = json.loads(lang_path.read_text())
        _supported_languages = {}
        for lang in raw_data:
            code = lang.get("code", "").lower()
            name = lang.get("name", "")
            if code and name:
                _supported_languages[code] = name
        # Ensure English is always supported
        if "en" not in _supported_languages:
            _supported_languages["en"] = "English"
        return _supported_languages
    except Exception:
        # Fallback to English only
        _supported_languages = {"en": "English"}
        return _supported_languages


def is_language_supported(lang_code: str) -> bool:
    """Check if a language code is in the supported languages list."""
    if not lang_code:
        return False
    supported = load_supported_languages()
    return lang_code.lower() in supported


def language_name_from_code(code: str | None) -> str:
    """Get language name from code, using supported languages list if available."""
    if not code:
        return "English"
    
    # First check supported languages
    supported = load_supported_languages()
    code_lower = code.lower()
    if code_lower in supported:
        return supported[code_lower]
    
    # Fallback to langcodes library
    try:
        return Language.get(code).language_name().title()
    except Exception:  # pragma: no cover - fallback for unknown codes
        if code_lower == "en":
            return "English"
        return code


async def ensure_vector_store() -> FishVectorStore:
    global vector_store
    if vector_store is not None:
        return vector_store

    data_path = settings.fish_data_path
    if not data_path.exists():
        raise RuntimeError(f"Fish data file not found at {data_path}")

    raw_data = json.loads(data_path.read_text())
    records = load_fish_records(raw_data)
    vector_store = FishVectorStore(
        client=client,
        records=records,
        embedding_model=settings.embedding_model,
    )
    await vector_store.build()
    return vector_store


async def ensure_whisper_model() -> whisper.Whisper:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = await asyncio.to_thread(
            whisper.load_model, settings.whisper_model
        )
    return _whisper_model


async def run_llm(messages: list[dict], temperature: float = 0.2) -> str:
    if hasattr(client, "responses"):
        response = await client.responses.create(
            model=settings.response_model,
            input=messages,
            temperature=temperature,
        )
        return extract_output_text(response)

    chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    response = await client.chat.completions.create(
        model=settings.response_model,
        messages=chat_messages,
        temperature=temperature,
    )
    return extract_chat_completion_text(response)


async def run_llm_stream(messages: list[dict], temperature: float = 0.2):
    """Stream LLM responses from OpenAI."""
    chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    
    stream = await client.chat.completions.create(
        model=settings.response_model,
        messages=chat_messages,
        temperature=temperature,
        stream=True,
    )
    
    async for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                yield delta.content


memory_store = ConversationMemory(
    session_factory=async_session_factory,
    client=client,
    run_llm=run_llm,
)


def is_listing_all_fishes_query(text: str) -> bool:
    """Detect if the query is asking for all available fishes."""
    text_lower = text.lower().strip()
    listing_patterns = [
        "what are the available",
        "what are available",
        "list all",
        "show me all",
        "tell me all",
        "what fishes",
        "what fish",
        "all fishes",
        "all fish",
        "available fishes",
        "available fish",
        "what do you have",
        "what do you sell",
        "what's available",
        "what is available",
        "show available",
        "list available",
        "tell me what",
        "what can i",
        "what can you",
    ]
    return any(pattern in text_lower for pattern in listing_patterns)


async def translate_text(text: str, target_language: str) -> str:
    if not text.strip():
        return ""
    prompt = (
        f"Translate the following text to {target_language}. "
        "Respond with only the translated sentence.\n\n"
        f"{text}"
    )
    messages = [
        {"role": "system", "content": "You are a precise translation assistant."},
        {"role": "user", "content": prompt},
    ]
    return await run_llm(messages, temperature=0.0)


async def generate_answer(
    *,
    catalog_context: list[str],
    english_text: str,
    language_name: str,
    language_code: str,
    conversation_summary: str | None,
    conversation_history: list[ConversationSnippet],
) -> str:
    catalog_block = "\n".join(f"- {item}" for item in catalog_context) or "No catalog data."
    
    # Use supported language name, or default to English if not supported
    if is_language_supported(language_code):
        target_language = language_name or "English"
    else:
        target_language = "English"
    conversation_segments: list[str] = []
    if conversation_summary:
        conversation_segments.append(f"Session summary:\n{conversation_summary}")
    if conversation_history:
        turns = "\n".join(
            f"{turn.role.title()}: {turn.content}" for turn in conversation_history
        )
        conversation_segments.append(f"Recent turns:\n{turns}")
    conversation_block = "\n\n".join(conversation_segments) or "No prior context."
    messages = [
        {
            "role": "system",
            "content": (
                "You are Meera, a helpful fish market seller. "
                "Use the provided fish catalog to answer questions about price, "
                "availability, and stock. Be concise, polite, and always end your "
                "reply by asking the customer how many kilograms they would like. "
                "If information is missing, say you can check with the supplier. "
                f"Replay answer in {target_language} language."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fish catalog:\n{catalog_block}\n\n"
                f"Conversation context:\n{conversation_block}\n\n"
                f"Customer request (English):\n{english_text}"
            ),
        },
    ]
    return await run_llm(messages, temperature=0.2)


async def generate_answer_stream(
    *,
    catalog_context: list[str],
    english_text: str,
    language_name: str,
    language_code: str,
    conversation_summary: str | None,
    conversation_history: list[ConversationSnippet],
):
    """Stream LLM responses."""
    catalog_block = "\n".join(f"- {item}" for item in catalog_context) or "No catalog data."
    
    # Use supported language name, or default to English if not supported
    if is_language_supported(language_code):
        target_language = language_name or "English"
    else:
        target_language = "English"
    conversation_segments: list[str] = []
    if conversation_summary:
        conversation_segments.append(f"Session summary:\n{conversation_summary}")
    if conversation_history:
        turns = "\n".join(
            f"{turn.role.title()}: {turn.content}" for turn in conversation_history
        )
        conversation_segments.append(f"Recent turns:\n{turns}")
    conversation_block = "\n\n".join(conversation_segments) or "No prior context."
    messages = [
        {
            "role": "system",
            "content": (
                "You are Meera, a helpful fish market seller. "
                "Use the provided fish catalog to answer questions about price, "
                "availability, and stock. Be concise, polite, and always end your "
                "reply by asking the customer how many kilograms they would like. "
                "If information is missing, say you can check with the supplier. "
                f"Replay answer in {target_language} language."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Fish catalog:\n{catalog_block}\n\n"
                f"Conversation context:\n{conversation_block}\n\n"
                f"Customer request (English):\n{english_text}"
            ),
        },
    ]
    async for chunk in run_llm_stream(messages, temperature=0.2):
        yield chunk


async def process_text_payload(text: str) -> tuple[str, str, str, str]:
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text input is empty.")

    # Common English words/phrases that might be misdetected
    common_english_phrases = [
        "i need", "i want", "how much", "what is", "tell me", "show me",
        "need", "want", "price", "cost", "available", "stock", "have",
        "do you have", "can you", "can i", "can i get", "i can", "i would like",
        "please", "thank you", "yes", "no", "give me", "get me", "i'll take",
        "i'd like", "how many", "how much", "what's the", "is there", "are there"
    ]
    
    # Common English greeting words (single words that are clearly English)
    common_english_words = [
        "hello", "hi", "hey", "hey there", "good morning", "good afternoon",
        "good evening", "good night", "thanks", "bye", "goodbye", "see you",
        "ok", "okay", "sure", "fine", "well", "okay", "alright", "right"
    ]
    
    # Check if text contains common English phrases (case insensitive)
    text_lower = cleaned.lower().strip()
    has_english_phrase = any(phrase in text_lower for phrase in common_english_phrases)
    
    # Check if text is a common English word
    is_common_english_word = text_lower in [w.lower() for w in common_english_words]
    
    # Check for English patterns: numbers with units (kg, kgs, pounds, etc.)
    has_english_patterns = bool(
        re.search(r'\d+\s*(kg|kgs|kilogram|kilograms|pound|pounds|lb|lbs|gram|grams|g)', text_lower, re.IGNORECASE) or
        re.search(r'\b(can|will|would|should|could|may|might)\s+(i|you|we|they)', text_lower) or
        re.search(r'\b(i|you|we|they)\s+(need|want|like|get|have|take)', text_lower) or
        re.search(r'\b(how|what|when|where|why|which)\s+', text_lower)
    )
    
    # For very short text or text with English phrases, be more conservative
    min_length_for_detection = 10
    is_short_text = len(cleaned.split()) < 3
    is_single_word = len(cleaned.split()) == 1
    
    lang_code = "en"  # Default to English
    confidence = 0.0
    
    # If it's a common English word, immediately default to English
    if is_common_english_word:
        lang_code = "en"
        lang_name = language_name_from_code(lang_code)
        return cleaned, cleaned, lang_code, lang_name
    
    try:
        # Try to detect with confidence scores
        detected_langs_list = detect_langs(cleaned)
        if detected_langs_list:
            top_lang = detected_langs_list[0]
            lang_code = top_lang.lang
            confidence = top_lang.prob
            
            # If text has English patterns (like "can i get", numbers with units), prioritize English
            if has_english_patterns and lang_code != "en":
                lang_code = "en"
            # For single words or very short text, require very high confidence for non-English
            elif is_single_word or is_short_text:
                # For single words, only trust non-English if confidence is extremely high
                min_confidence = 0.95 if is_single_word else 0.8
                if lang_code != "en" and confidence < min_confidence:
                    lang_code = "en"
                # If it's a single word and not English, be very cautious
                elif is_single_word and lang_code != "en":
                    # Check if it looks like English (ASCII characters, common patterns)
                    if cleaned.isascii() and not any(char in cleaned for char in "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"):
                        lang_code = "en"
            else:
                # For longer text, use normal confidence thresholds
                # But if it has English phrases or patterns, require higher confidence for non-English
                min_confidence = 0.8 if (has_english_phrase or has_english_patterns) else 0.5
                
                # If confidence is low or text seems English, default to English
                if confidence < min_confidence or (has_english_phrase and lang_code != "en"):
                    lang_code = "en"
                # If text has English patterns but detected as non-English, override
                elif has_english_patterns and lang_code != "en":
                    lang_code = "en"
                # If detected language is not English but confidence is very high, trust it
                elif lang_code != "en" and confidence >= 0.9 and not has_english_patterns:
                    # Trust the detection only if no English patterns found
                    pass
                # For ambiguous cases with English phrases, default to English
                elif has_english_phrase:
                    lang_code = "en"
    except LangDetectException:
        lang_code = "en"
    except Exception:
        # Fallback to simple detect if detect_langs fails
        try:
            lang_code = detect(cleaned)
            # If it's not English but text has English phrases, patterns, or is a common word, override
            if (has_english_phrase or has_english_patterns or is_common_english_word) and lang_code != "en":
                lang_code = "en"
            # For single words, be cautious
            if is_single_word and lang_code != "en" and cleaned.isascii():
                lang_code = "en"
        except LangDetectException:
            lang_code = "en"
    
    # Check if detected language is supported
    if not is_language_supported(lang_code):
        # If not supported, default to English
        lang_code = "en"
        lang_name = "English"
        return cleaned, cleaned, lang_code, lang_name
    
    lang_name = language_name_from_code(lang_code)

    if lang_code.lower() == "en":
        return cleaned, cleaned, lang_code, lang_name

    english_text = await translate_text(cleaned, "English")
    return cleaned, english_text, lang_code, lang_name


async def upload_audio_to_s3(
    audio_blob: bytes, filename: str | None, content_type: str | None, session_id: str
) -> str:
    """Upload audio file to S3 and return the public URL."""
    if not audio_blob:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # Generate unique S3 key
    file_extension = Path(filename or "audio.wav").suffix or ".wav"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    s3_key = f"audio/{session_id}/{timestamp}_{unique_id}{file_extension}"

    # Upload to S3
    await asyncio.to_thread(
        s3_client.put_object,
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
        Body=audio_blob,
        ContentType=content_type or "audio/wav",
    )

    # Generate public URL
    s3_url = f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"
    return s3_url


async def process_audio_payload(audio_blob: bytes, filename: str | None) -> tuple[str, str, str, str]:
    """Process audio blob for transcription."""
    model = await ensure_whisper_model()
    if not audio_blob:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    suffix = Path(filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_blob)
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(
            model.transcribe,
            tmp_path,
            task="translate",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]

    english_text = result.get("text", "").strip()
    if not english_text:
        raise HTTPException(status_code=400, detail="Unable to transcribe audio.")

    lang_code = result.get("language", "en")
    
    # Check if detected language is supported
    if not is_language_supported(lang_code):
        # If not supported, default to English
        lang_code = "en"
    
    lang_name = language_name_from_code(lang_code)
    return english_text, english_text, lang_code, lang_name


@app.on_event("startup")
async def startup_event() -> None:
    await asyncio.gather(ensure_vector_store(), ensure_whisper_model())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/chat/history")
async def get_chat_history(
    session_id: str = Query(..., description="Session identifier to retrieve chat history"),
    limit: int | None = Query(default=None, ge=1, le=1000, description="Maximum number of messages to retrieve"),
) -> dict[str, Any]:
    """Retrieve full conversation history for a given session_id."""
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")
    
    if memory_store is None:
        raise HTTPException(status_code=500, detail="Memory store unavailable.")
    
    history = await memory_store.get_history(session_id=session_id.strip(), limit=limit)
    
    return {
        "session_id": session_id.strip(),
        "message_count": len(history),
        "messages": history,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["session_id"],
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session identifier for conversation memory",
                            },
                            "text": {
                                "type": "string",
                                "description": "Customer text message (optional if audio is provided)",
                            },
                            "audio": {
                                "type": "string",
                                "format": "binary",
                                "description": "Customer audio message (optional if text is provided)",
                            },
                        },
                    },
                },
            },
        },
    },
)
async def chat_endpoint(
    request: Request,
    session_id: str = Form(
        ...,
        description="Session identifier for conversation memory",
    ),
    text: str | None = Form(
        default=None,
        description="Customer text message (optional if audio is provided)",
    ),
    audio: UploadFile | None = File(  # pyright: ignore[reportUndefinedVariable]
        default=None,
        description="Customer audio message (optional if text is provided)",
    ),
) -> ChatResponse:
    session_id = session_id.strip()
    # Handle text - if it's empty string, treat as None
    text_value = text.strip() if (text and isinstance(text, str) and text.strip()) else None

    # Handle audio file - check if it's a valid UploadFile
    audio_file: UploadFile | None = None
    if audio is not None:
        # Check if it's actually a valid file (not empty string or invalid object)
        try:
            # Try to access UploadFile attributes to verify it's real
            if hasattr(audio, 'filename') or hasattr(audio, 'content_type') or hasattr(audio, 'read'):
                # It has UploadFile attributes, accept it
                # We'll verify it has content when reading
                audio_file = audio
        except (AttributeError, TypeError):
            # If it's not a valid UploadFile, ignore it
            pass
    
    # If audio parameter failed validation but might be in form, try manual parsing
    if audio_file is None:
        try:
            form = await request.form()
            audio_candidate = form.get("audio")
            if audio_candidate and isinstance(audio_candidate, UploadFile):
                audio_file = audio_candidate
        except Exception:
            pass

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required to maintain context.",
        )

    # Validate that at least one input is provided
    if text_value is None and audio_file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a non-empty 'text' field or an 'audio' file.",
        )

    audio_url: str | None = None
    if audio_file is not None:
        source: Literal["text", "audio"] = "audio"
        # Read audio file once
        audio_blob = await audio_file.read()
        if not audio_blob:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
        
        # Upload audio to S3
        audio_url = await upload_audio_to_s3(
            audio_blob=audio_blob,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            session_id=session_id,
        )
        
        # Process audio for transcription
        (
            original_text,
            english_text,
            detected_code,
            language_name,
        ) = await process_audio_payload(audio_blob, audio_file.filename)
    else:
        source = "text"
        (
            original_text,
            english_text,
            detected_code,
            language_name,
        ) = await process_text_payload(text_value or "")

    store = await ensure_vector_store()
    # If query is asking for all available fishes, return all records
    if is_listing_all_fishes_query(english_text):
        catalog_context = store.get_all_records()
    else:
        catalog_context = await store.query(english_text, settings.max_context_items)

    if memory_store is None:
        raise HTTPException(status_code=500, detail="Memory store unavailable.")

    conversation_summary, conversation_history = await memory_store.build_context(
        session_id=session_id,
        query_text=english_text,
    )

    seller_reply = await generate_answer(
        catalog_context=catalog_context,
        english_text=english_text,
        language_name=language_name,
        language_code=detected_code,
        conversation_summary=conversation_summary,
        conversation_history=conversation_history,
    )

    # Determine reply language based on supported languages
    if is_language_supported(detected_code) and language_name.lower() != "english":
        # Reply in detected language if supported
        seller_reply_en = await translate_text(seller_reply, "English")
        final_reply = seller_reply
    else:
        # Reply in English (either detected as English or unsupported language)
        seller_reply_en = seller_reply
        final_reply = seller_reply

    await memory_store.record_message(
        session_id=session_id,
        role="user",
        content=english_text,
        metadata={
            "original_text": original_text,
            "language": language_name,
            "source": source,
            "audio_url": audio_url,
        },
        trigger_summary=False,
    )

    await memory_store.record_message(
        session_id=session_id,
        role="assistant",
        content=seller_reply_en,
        metadata={
            "reply_language": language_name,
            "translated_reply": final_reply if language_name.lower() != "english" else None,
        },
        trigger_summary=True,
    )

    return ChatResponse(
        detected_language=language_name,
        detected_language_code=detected_code,
        source=source,
        customer_text=original_text,
        english_text=english_text,
        seller_reply_english=seller_reply_en,
        seller_reply=final_reply,
        context_used=catalog_context,
        audio_url=audio_url,
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: Request,
    session_id: str = Form(
        ...,
        description="Session identifier for conversation memory",
    ),
    text: str | None = Form(
        default=None,
        description="Customer text message (optional if audio is provided)",
    ),
    audio: UploadFile | None = File(  # pyright: ignore[reportUndefinedVariable]
        default=None,
        description="Customer audio message (optional if text is provided)",
    ),
):
    """Stream chat responses from OpenAI."""
    session_id = session_id.strip()
    text_value = text.strip() if (text and isinstance(text, str) and text.strip()) else None

    # Handle audio file
    audio_file: UploadFile | None = None
    if audio is not None:
        try:
            if hasattr(audio, 'filename') or hasattr(audio, 'content_type') or hasattr(audio, 'read'):
                audio_file = audio
        except (AttributeError, TypeError):
            pass
    
    if audio_file is None:
        try:
            form = await request.form()
            audio_candidate = form.get("audio")
            if audio_candidate and isinstance(audio_candidate, UploadFile):
                audio_file = audio_candidate
        except Exception:
            pass

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required to maintain context.")

    if text_value is None and audio_file is None:
        raise HTTPException(
            status_code=400,
            detail="Provide either a non-empty 'text' field or an 'audio' file.",
        )

    audio_url: str | None = None
    if audio_file is not None:
        source: Literal["text", "audio"] = "audio"
        audio_blob = await audio_file.read()
        if not audio_blob:
            raise HTTPException(status_code=400, detail="Audio file is empty.")
        
        audio_url = await upload_audio_to_s3(
            audio_blob=audio_blob,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
            session_id=session_id,
        )
        
        (
            original_text,
            english_text,
            detected_code,
            language_name,
        ) = await process_audio_payload(audio_blob, audio_file.filename)
    else:
        source = "text"
        (
            original_text,
            english_text,
            detected_code,
            language_name,
        ) = await process_text_payload(text_value or "")

    store = await ensure_vector_store()
    # If query is asking for all available fishes, return all records
    if is_listing_all_fishes_query(english_text):
        catalog_context = store.get_all_records()
    else:
        catalog_context = await store.query(english_text, settings.max_context_items)

    if memory_store is None:
        raise HTTPException(status_code=500, detail="Memory store unavailable.")

    conversation_summary, conversation_history = await memory_store.build_context(
        session_id=session_id,
        query_text=english_text,
    )

    async def generate_stream():
        """Generate streaming response with metadata and text chunks."""
        # Send metadata first
        metadata = {
            "type": "metadata",
            "detected_language": language_name,
            "detected_language_code": detected_code,
            "source": source,
            "customer_text": original_text,
            "english_text": english_text,
            "audio_url": audio_url,
        }
        yield f"data: {json.dumps(metadata)}\n\n"
        
        # Stream the response
        full_reply = ""
        async for chunk in generate_answer_stream(
            catalog_context=catalog_context,
            english_text=english_text,
            language_name=language_name,
            language_code=detected_code,
            conversation_summary=conversation_summary,
            conversation_history=conversation_history,
        ):
            full_reply += chunk
            # Send each chunk
            chunk_data = {
                "type": "chunk",
                "content": chunk,
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send completion signal
        completion_data = {
            "type": "done",
            "full_reply": full_reply,
        }
        yield f"data: {json.dumps(completion_data)}\n\n"
        
        # Save to memory after streaming
        # Determine reply language based on supported languages
        if is_language_supported(detected_code) and language_name.lower() != "english":
            # Reply in detected language if supported
            seller_reply_en = await translate_text(full_reply, "English")
            final_reply = full_reply
        else:
            # Reply in English (either detected as English or unsupported language)
            seller_reply_en = full_reply
            final_reply = full_reply

        await memory_store.record_message(
            session_id=session_id,
            role="user",
            content=english_text,
            metadata={
                "original_text": original_text,
                "language": language_name,
                "source": source,
                "audio_url": audio_url,
            },
            trigger_summary=False,
        )

        await memory_store.record_message(
            session_id=session_id,
            role="assistant",
            content=seller_reply_en,
            metadata={
                "reply_language": language_name,
                "translated_reply": final_reply if language_name.lower() != "english" else None,
            },
            trigger_summary=True,
        )

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

