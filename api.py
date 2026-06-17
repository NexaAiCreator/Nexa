import argparse
import asyncio
import base64
from collections import deque
import ctypes
import io
import json
import logging
import os
import re
import tempfile
import time
import wave
from contextlib import asynccontextmanager
from threading import Thread

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from pydantic import BaseModel, Field
from starlette.responses import Response, StreamingResponse
import uvicorn

from app_config import load_profile
from qwen_model import DEFAULT_MODEL, generate_reply, load_model, stream_reply
from agent_service import PlanningEngine, ObservationAnalyzer, ResultSynthesizer

load_dotenv(override=True)
logger = logging.getLogger(__name__)


def get_allowed_origins() -> list[str]:
    defaults = [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    raw = os.getenv("NEXA_ALLOWED_ORIGINS", "")
    configured = [origin.strip() for origin in raw.split(",") if origin.strip()]

    seen = set()
    origins: list[str] = []
    for origin in [*defaults, *configured]:
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class BrowserPageContext(BaseModel):
    url: str | None = None
    title: str | None = None
    content: str | None = None
    selection: str | None = None
    content_truncated: bool = False


class BrowserOpenTab(BaseModel):
    title: str | None = None
    url: str | None = None
    kind: str | None = None


class BrowserMemoryItem(BaseModel):
    type: str | None = None
    title: str | None = None
    url: str | None = None


class BrowserContext(BaseModel):
    page: BrowserPageContext | None = None
    open_tabs: list[BrowserOpenTab] = Field(default_factory=list)
    memory: list[BrowserMemoryItem] = Field(default_factory=list)


class BrowserClientInfo(BaseModel):
    name: str | None = None
    version: str | None = None
    platform: str | None = None


class BrowserGenerationOptions(BaseModel):
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=300, ge=1, le=2048)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class BrowserExecuteRequest(BaseModel):
    version: str = "1.0"
    request_id: str | None = None
    action: str
    user_prompt: str | None = None
    context: BrowserContext = Field(default_factory=BrowserContext)
    permissions: dict[str, bool] = Field(default_factory=dict)
    client: BrowserClientInfo | None = None
    generation: BrowserGenerationOptions = Field(default_factory=BrowserGenerationOptions)


# Agent planning models
class AgentContext(BaseModel):
    current_url: str | None = None
    current_title: str | None = None
    open_tabs: list[dict] = Field(default_factory=list)
    bookmarks: list[dict] = Field(default_factory=list)
    available_permissions: list[str] = Field(default_factory=list)


class AgentPlanRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    context: AgentContext = Field(default_factory=AgentContext)
    max_steps: int = Field(default=20, ge=1, le=50)


class WorkflowStep(BaseModel):
    id: str
    type: str  # "action", "decision", "clarification", "synthesize"
    content: dict
    condition: str | None = None


class AgentPlanResponse(BaseModel):
    workflow_id: str
    goal: str
    description: str
    steps: list[WorkflowStep]
    required_permissions: list[str]
    risk_assessment: dict
    estimated_duration_seconds: int
    created_at: str


class Observation(BaseModel):
    action_id: str
    workflow_id: str
    type: str  # action type
    success: bool
    status_code: int | None = None
    result: dict | None = None
    timing: dict | None = None
    dom_state: dict | None = None
    error: str | None = None


class AgentExecuteRequest(BaseModel):
    workflow_id: str
    step_id: str
    action_type: str
    params: dict
    screenshot: bool = False


class AgentExecuteResponse(BaseModel):
    step_id: str
    action_id: str
    success: bool
    observation: dict
    next_step_id: str | None = None


class ClarificationResponse(BaseModel):
    workflow_id: str
    step_id: str
    response: str | None = None
    approval: bool | None = None


class ObservationRequest(BaseModel):
    workflow_id: str
    observations: list[Observation]


class ObservationResponse(BaseModel):
    workflow_id: str
    analysis: dict
    needs_adaptation: bool
    suggested_actions: list[str]


class SttResponse(BaseModel):
    text: str


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str | None = None
    rate: str | None = None
    voice_id: str | None = None
    model_id: str | None = None


class WakeDetectRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    audio_b64: str | None = None
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    reset: bool = False


class WakeDetectResponse(BaseModel):
    detected: bool
    score: float = 0.0
    threshold: float = 0.5
    supported: bool = False
    model: str | None = None


def build_chat_messages(
    incoming_messages: list[dict],
    fallback_system_prompt: str,
) -> list[dict]:
    messages = [item for item in incoming_messages if item.get("content")]
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": fallback_system_prompt}, *messages]
    return messages


BROWSER_ACTION_SPECS = {
    "summarize_page": {
        "result_type": "summary",
        "required_permissions": ["current_page"],
        "requires_page_content": True,
    },
    "summarize_selection": {
        "result_type": "summary",
        "required_permissions": ["selected_text"],
        "requires_selection": True,
    },
    "answer_with_page_context": {
        "result_type": "answer",
        "required_permissions": ["current_page"],
        "requires_page_content": True,
        "requires_user_prompt": True,
    },
    "answer_with_browser_context": {
        "result_type": "answer",
        "required_permissions": [],
        "requires_user_prompt": True,
    },
    "rewrite_selection": {
        "result_type": "rewrite",
        "required_permissions": ["selected_text"],
        "requires_selection": True,
    },
}


def make_browser_error(request_id: str | None, action: str, code: str, message: str) -> dict:
    return {
        "ok": False,
        "request_id": request_id,
        "action": action,
        "error": {
            "code": code,
            "message": message,
        },
    }


def get_browser_capabilities(assistant_name: str, model_name: str, adapter_dir: str | None) -> dict:
    return {
        "version": "1.0",
        "assistant_name": assistant_name,
        "model": {
            "name": model_name,
            "adapter_dir": adapter_dir,
        },
        "supports": {
            "execute": True,
            "stream": True,
            "json_mode": True,
            "stt": True,
            "tts": True,
        },
        "actions": list(BROWSER_ACTION_SPECS.keys()),
        "limits": {
            "max_page_chars": 6000,
            "max_selection_chars": 2000,
            "max_memory_items": 20,
            "max_open_tabs": 25,
        },
        "voice": {
            "provider": get_tts_provider(),
            "default_voice": get_default_tts_voice(),
            "format": "audio/mpeg",
        },
    }


def validate_browser_request(request: BrowserExecuteRequest) -> tuple[bool, dict | None]:
    spec = BROWSER_ACTION_SPECS.get(request.action)
    if not spec:
        return False, make_browser_error(
            request.request_id,
            request.action,
            "UNKNOWN_ACTION",
            f"Unsupported browser action: {request.action}",
        )

    for scope in spec["required_permissions"]:
        if not request.permissions.get(scope, False):
            return False, make_browser_error(
                request.request_id,
                request.action,
                "PERMISSION_DENIED",
                f"Permission denied for scope: {scope}",
            )

    page = request.context.page
    if spec.get("requires_page_content") and not (page and page.content and page.url):
        return False, make_browser_error(
            request.request_id,
            request.action,
            "INVALID_CONTEXT",
            f"Page content is required for {request.action}.",
        )

    if spec.get("requires_selection") and not (page and page.selection and page.selection.strip()):
        return False, make_browser_error(
            request.request_id,
            request.action,
            "INVALID_CONTEXT",
            f"Selected text is required for {request.action}.",
        )

    if spec.get("requires_user_prompt") and not (request.user_prompt and request.user_prompt.strip()):
        return False, make_browser_error(
            request.request_id,
            request.action,
            "INVALID_REQUEST",
            f"user_prompt is required for {request.action}.",
        )

    if request.action == "answer_with_browser_context":
        has_any_context = bool(
            (page and ((page.content and page.content.strip()) or (page.selection and page.selection.strip())))
            or request.context.open_tabs
            or request.context.memory
        )
        if not has_any_context:
            return False, make_browser_error(
                request.request_id,
                request.action,
                "INVALID_CONTEXT",
                "At least one browser context source is required for answer_with_browser_context.",
            )

    return True, None


def sanitize_browser_context(request: BrowserExecuteRequest) -> dict:
    page = request.context.page
    page_payload = None
    if page:
        page_payload = {
            "url": page.url,
            "title": page.title,
            "content": page.content,
            "selection": page.selection,
            "content_truncated": page.content_truncated,
        }

    return {
        "page": page_payload,
        "open_tabs": [
            {"title": item.title, "url": item.url, "kind": item.kind}
            for item in request.context.open_tabs[:25]
        ],
        "memory": [
            {"type": item.type, "title": item.title, "url": item.url}
            for item in request.context.memory[:20]
        ],
    }


def build_browser_messages(request: BrowserExecuteRequest, fallback_system_prompt: str) -> list[dict]:
    spec = BROWSER_ACTION_SPECS[request.action]
    context_payload = sanitize_browser_context(request)
    instructions = {
        "summarize_page": "Summarize the provided page clearly and concisely.",
        "summarize_selection": "Summarize the selected text only.",
        "answer_with_page_context": "Answer the user's prompt using the provided page context.",
        "answer_with_browser_context": "Answer the user's prompt using only the provided browser context.",
        "rewrite_selection": "Rewrite the selected text according to the user's instruction. If no instruction is given, improve clarity while preserving intent.",
    }
    schema_hint = {
        "type": spec["result_type"],
        "text": "...",
    }

    system_content = (
        f"{fallback_system_prompt}\n\n"
        "You are operating as the Nexa Browser AI service.\n"
        "Only use the context provided in the request. Do not assume access to hidden browser data.\n"
        "Return valid JSON only with this top-level shape:\n"
        '{'
        '"type": string, '
        '"text": string, '
        '"title": string|null, '
        '"source": {"url": string|null}|null, '
        '"used_context": {"page": boolean, "open_tabs": boolean, "memory_items": number}|null, '
        '"warnings": string[]'
        '}\n'
        f"Set at least these fields: {json.dumps(schema_hint)}"
    )

    user_payload = {
        "action": request.action,
        "instruction": instructions[request.action],
        "user_prompt": request.user_prompt,
        "permissions": request.permissions,
        "context": context_payload,
    }

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def build_browser_stream_messages(request: BrowserExecuteRequest, fallback_system_prompt: str) -> list[dict]:
    context_payload = sanitize_browser_context(request)
    instructions = {
        "answer_with_page_context": (
            "Answer the user's prompt using only the provided page context. "
            "Respond in plain text only. Do not return JSON, code fences, XML, or metadata labels."
        ),
        "answer_with_browser_context": (
            "Answer the user's prompt using only the provided browser context. "
            "Respond in plain text only. Do not return JSON, code fences, XML, or metadata labels."
        ),
    }

    system_content = (
        f"{fallback_system_prompt}\n\n"
        "You are operating as the Nexa Browser AI service.\n"
        "Only use the context provided in the request. Do not assume access to hidden browser data.\n"
        "For this route, return plain text only.\n"
        "Do not output JSON.\n"
        "Do not include field names like type, text, source, warnings, or used_context.\n"
        "If the page context is insufficient, say that plainly and briefly."
    )

    user_payload = {
        "action": request.action,
        "instruction": instructions.get(request.action, "Answer using only the provided browser context."),
        "user_prompt": request.user_prompt,
        "permissions": request.permissions,
        "context": context_payload,
    }

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def parse_model_json(text: str) -> dict:
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        candidate = next((part for part in parts if "{" in part and "}" in part), candidate)
        candidate = candidate.replace("json", "", 1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def normalize_browser_result(request: BrowserExecuteRequest, parsed: dict) -> dict:
    spec = BROWSER_ACTION_SPECS[request.action]
    result_type = parsed.get("type") or spec["result_type"]
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model response JSON did not include a non-empty text field.")

    page = request.context.page
    source_url = None
    if isinstance(parsed.get("source"), dict):
        source_url = parsed["source"].get("url")
    if not source_url and page and page.url:
        source_url = page.url

    if request.action == "answer_with_browser_context":
        default_used_context = {
            "page": bool(page and ((page.content and page.content.strip()) or (page.selection and page.selection.strip()))),
            "open_tabs": bool(request.context.open_tabs),
            "memory_items": len(request.context.memory),
        }
    else:
        default_used_context = None

    return {
        "type": result_type,
        "text": text.strip(),
        "title": parsed.get("title") if isinstance(parsed.get("title"), str) else (page.title if page else None),
        "source": {"url": source_url} if source_url else None,
        "used_context": parsed.get("used_context") if isinstance(parsed.get("used_context"), dict) else default_used_context,
        "warnings": parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else [],
    }


def normalize_browser_fallback_result(request: BrowserExecuteRequest, raw_reply: str, warning: str) -> dict:
    spec = BROWSER_ACTION_SPECS[request.action]
    page = request.context.page
    text = raw_reply.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    text_match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if text_match:
        extracted = text_match.group(1)
        text = bytes(extracted, "utf-8").decode("unicode_escape").strip()

    if request.action == "answer_with_browser_context":
        used_context = {
            "page": bool(page and ((page.content and page.content.strip()) or (page.selection and page.selection.strip()))),
            "open_tabs": bool(request.context.open_tabs),
            "memory_items": len(request.context.memory),
        }
    else:
        used_context = None

    return {
        "type": spec["result_type"],
        "text": text,
        "title": page.title if page else None,
        "source": {"url": page.url} if page and page.url else None,
        "used_context": used_context,
        "warnings": [warning],
    }


def get_stt_model(app: FastAPI):
    stt_model = getattr(app.state, "stt_model", None)
    if stt_model is not None:
        return stt_model

    from faster_whisper import WhisperModel

    model_size = os.getenv("NEXA_STT_MODEL", "tiny")
    requested_device = os.getenv("NEXA_STT_DEVICE", "auto").strip().lower() or "auto"
    device = requested_device
    compute_type = os.getenv("NEXA_STT_COMPUTE_TYPE", "int8")
    strict_gpu = requested_device == "cuda"

    if os.name == "nt":
        import site
        nvidia_bin_dirs: list[str] = []
        for package_root in site.getsitepackages():
            nvidia_root = os.path.join(package_root, "nvidia")
            if not os.path.isdir(nvidia_root):
                continue
            for child in os.listdir(nvidia_root):
                bin_dir = os.path.join(nvidia_root, child, "bin")
                if os.path.isdir(bin_dir):
                    nvidia_bin_dirs.append(bin_dir)
        for dll_dir in nvidia_bin_dirs:
            try:
                os.add_dll_directory(dll_dir)
            except Exception:
                pass

    if requested_device in {"auto", "cuda"} and os.name == "nt":
        try:
            ctypes.WinDLL("cublas64_12.dll")
        except OSError:
            if strict_gpu:
                raise RuntimeError(
                    "NEXA_STT_DEVICE=cuda but cublas64_12.dll is missing. "
                    "Install CUDA runtime dependencies in the venv."
                )
            device = "cpu"
            logger.warning(
                "CUDA STT runtime not available (cublas64_12.dll missing). "
                "Falling back to CPU for faster and stable transcription."
            )
    try:
        app.state.stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        app.state.stt_model_runtime = {
            "model_size": model_size,
            "device": device,
            "compute_type": compute_type,
        }
        return app.state.stt_model
    except Exception:
        if strict_gpu:
            raise
        fallback_device = "cpu"
        fallback_compute_type = "int8"
        app.state.stt_model = WhisperModel(
            model_size,
            device=fallback_device,
            compute_type=fallback_compute_type,
        )
        app.state.stt_model_runtime = {
            "model_size": model_size,
            "device": fallback_device,
            "compute_type": fallback_compute_type,
        }
        return app.state.stt_model


def get_stt_language() -> str:
    return os.getenv("NEXA_STT_LANGUAGE", "en").strip() or "en"


def get_default_tts_voice() -> str:
    return os.getenv("NEXA_TTS_VOICE", "en-US-AvaMultilingualNeural")


def get_tts_provider() -> str:
    return os.getenv("NEXA_TTS_PROVIDER", "edge").strip().lower() or "edge"


def normalize_tts_rate(rate: str | None) -> str:
    raw = (rate or os.getenv("NEXA_TTS_RATE", "+0%")).strip()
    if not raw:
        return "+0%"
    if re.fullmatch(r"[+-]?\d+%", raw):
        if raw.startswith(("+", "-")):
            return raw
        return f"+{raw}"
    return "+0%"


def get_elevenlabs_defaults() -> dict:
    return {
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "").strip(),
        "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip() or "eleven_flash_v2_5",
        "output_format": os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip() or "mp3_44100_128",
        "api_key": os.getenv("ELEVENLABS_API_KEY", "").strip(),
    }


def get_wake_threshold() -> float:
    raw = os.getenv("NEXA_WAKE_THRESHOLD", "0.35").strip() or "0.35"
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.35


def get_wake_consecutive_hits() -> int:
    raw = os.getenv("NEXA_WAKE_CONSECUTIVE_HITS", "1").strip() or "1"
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 1


def get_wake_score_window() -> int:
    raw = os.getenv("NEXA_WAKE_SCORE_WINDOW", "4").strip() or "4"
    try:
        return max(1, min(12, int(raw)))
    except ValueError:
        return 4


def get_wake_cooldown_seconds() -> float:
    raw = os.getenv("NEXA_WAKE_COOLDOWN_SECONDS", "1.5").strip() or "1.5"
    try:
        return max(0.0, min(10.0, float(raw)))
    except ValueError:
        return 1.5


def get_wake_model_path() -> str:
    return os.getenv("NEXA_WAKE_MODEL_PATH", "").strip()


def is_wake_detection_enabled() -> bool:
    return bool(get_wake_model_path())


def create_wake_detector():
    model_path = get_wake_model_path()
    if not model_path:
        raise RuntimeError("NEXA_WAKE_MODEL_PATH is not configured.")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Wake model not found at {model_path}")

    from openwakeword.model import Model

    return {
        "target_model": os.path.splitext(os.path.basename(model_path))[0],
        "model": Model(
            wakeword_models=[model_path],
            vad_threshold=0.0,
        ),
        "recent_scores": deque(maxlen=get_wake_score_window()),
        "consecutive_hits": 0,
        "last_detection_at": 0.0,
        "updated_at": time.time(),
    }


def get_wake_detectors(app: FastAPI) -> dict[str, dict]:
    detectors = getattr(app.state, "wake_detectors", None)
    if detectors is None:
        detectors = {}
        app.state.wake_detectors = detectors
    return detectors


def reset_wake_detector(app: FastAPI, session_id: str) -> None:
    get_wake_detectors(app).pop(session_id, None)


def get_wake_detector(app: FastAPI, session_id: str) -> dict:
    detectors = get_wake_detectors(app)
    detector = detectors.get(session_id)
    if detector is None:
        detector = create_wake_detector()
        detectors[session_id] = detector
    detector["updated_at"] = time.time()
    return detector


async def synthesize_edge_tts_audio(text: str, voice: str | None, rate: str | None) -> tuple[bytes, str]:
    import edge_tts

    selected_voice = (voice or get_default_tts_voice()).strip() or get_default_tts_voice()
    selected_rate = normalize_tts_rate(rate)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        communicator = edge_tts.Communicate(text=text, voice=selected_voice, rate=selected_rate)
        await communicator.save(temp_path)
        with open(temp_path, "rb") as handle:
            return handle.read(), "audio/mpeg"
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def synthesize_elevenlabs_tts_audio(
    text: str,
    voice_id: str | None,
    model_id: str | None,
) -> tuple[bytes, str]:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    defaults = get_elevenlabs_defaults()
    api_key = defaults["api_key"]
    selected_voice_id = (voice_id or defaults["voice_id"]).strip()
    selected_model_id = (model_id or defaults["model_id"]).strip()
    output_format = defaults["output_format"]

    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured.")
    if not selected_voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not configured.")

    client = ElevenLabs(api_key=api_key)
    
    # Parse output format to extract quality settings
    # Format: mp3_44100_128 (format_samplerate_bitrate)
    voice_settings = VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        use_speaker_boost=False,
    )

    response = client.text_to_speech.convert(
        voice_id=selected_voice_id,
        text=text,
        model_id=selected_model_id,
        output_format=output_format,
        voice_settings=voice_settings,
    )
    
    # Collect all chunks from the generator
    audio_bytes = b""
    for chunk in response:
        if chunk:
            audio_bytes += chunk
    
    return audio_bytes, "audio/mpeg"


async def synthesize_tts_audio(
    text: str,
    voice: str | None,
    rate: str | None,
    voice_id: str | None,
    model_id: str | None,
) -> tuple[bytes, str]:
    provider = get_tts_provider()
    if provider == "elevenlabs":
        return await synthesize_elevenlabs_tts_audio(text=text, voice_id=voice_id, model_id=model_id)
    return await synthesize_edge_tts_audio(text=text, voice=voice, rate=rate)


def build_app(model_name: str, adapter_dir: str | None, system_prompt: str, assistant_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tokenizer, model = load_model(model_name, adapter_dir)
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.stt_model = None

        if os.getenv("NEXA_PRELOAD_STT", "1").strip().lower() in {"1", "true", "yes", "on"}:
            def preload_stt():
                try:
                    get_stt_model(app)
                except Exception:
                    pass

            Thread(target=preload_stt, daemon=True).start()
        yield

    app = FastAPI(
        title=f"{assistant_name} API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.system_prompt = system_prompt
    app.state.assistant_name = assistant_name
    app.state.model_name = model_name
    app.state.adapter_dir = adapter_dir

    @app.get("/health")
    def health():
        import torch

        from qwen_model import get_generation_device

        runtime: dict = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
        }
        try:
            import bitsandbytes

            runtime["bitsandbytes_version"] = bitsandbytes.__version__
        except ImportError:
            runtime["bitsandbytes_version"] = None

        model = getattr(app.state, "model", None)
        if model is not None:
            try:
                runtime["model_device"] = str(get_generation_device(model))
            except Exception:
                runtime["model_device"] = "unknown"
        else:
            runtime["model_device"] = None

        return {
            "status": "ok",
            "assistant_name": app.state.assistant_name,
            "model": app.state.model_name,
            "adapter_dir": app.state.adapter_dir,
            "runtime": runtime,
            "stt_runtime": getattr(app.state, "stt_model_runtime", None),
        }

    @app.get("/ui-config")
    def ui_config():
        return {
            "assistant_name": app.state.assistant_name,
            "system_prompt": app.state.system_prompt,
        }

    @app.get("/v1/browser/capabilities")
    def browser_capabilities():
        return get_browser_capabilities(
            assistant_name=app.state.assistant_name,
            model_name=app.state.model_name,
            adapter_dir=app.state.adapter_dir,
        )

    @app.post("/generate")
    def generate(request: GenerateRequest):
        messages = [
            {"role": "system", "content": request.system_prompt or app.state.system_prompt},
            {"role": "user", "content": request.prompt},
        ]
        reply = generate_reply(
            tokenizer=app.state.tokenizer,
            model=app.state.model,
            messages=messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        return {"reply": reply}

    @app.post("/chat")
    def chat(request: ChatRequest):
        messages = build_chat_messages(
            [{"role": item.role, "content": item.content} for item in request.messages],
            app.state.system_prompt,
        )
        reply = generate_reply(
            tokenizer=app.state.tokenizer,
            model=app.state.model,
            messages=messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        return {"reply": reply}

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        messages = build_chat_messages(
            [{"role": item.role, "content": item.content} for item in request.messages],
            app.state.system_prompt,
        )

        def event_stream():
            for chunk in stream_reply(
                tokenizer=app.state.tokenizer,
                model=app.state.model,
                messages=messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            ):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")

    @app.post("/v1/browser/execute")
    def browser_execute(request: BrowserExecuteRequest):
        is_valid, error = validate_browser_request(request)
        if not is_valid:
            return error

        messages = build_browser_messages(request, app.state.system_prompt)
        reply = generate_reply(
            tokenizer=app.state.tokenizer,
            model=app.state.model,
            messages=messages,
            max_new_tokens=request.generation.max_new_tokens,
            temperature=request.generation.temperature,
            top_p=request.generation.top_p,
            top_k=request.generation.top_k,
            repetition_penalty=request.generation.repetition_penalty,
        )

        try:
            parsed = parse_model_json(reply)
            result = normalize_browser_result(request, parsed)
        except Exception as exc:
            result = normalize_browser_fallback_result(
                request,
                reply,
                f"Model returned unstructured output; browser used fallback parsing ({exc}).",
            )

        return {
            "ok": True,
            "request_id": request.request_id,
            "action": request.action,
            "result": result,
            "warnings": result.get("warnings", []),
            "model": {
                "name": app.state.model_name,
                "adapter_dir": app.state.adapter_dir,
            },
        }

    @app.post("/v1/browser/stream")
    def browser_stream(request: BrowserExecuteRequest):
        is_valid, error = validate_browser_request(request)
        if not is_valid:
            def error_stream():
                yield f"[error] {error['error']['message']}"
            return StreamingResponse(error_stream(), media_type="text/plain; charset=utf-8")

        messages = build_browser_stream_messages(request, app.state.system_prompt)

        def answer_stream():
            for chunk in stream_reply(
                tokenizer=app.state.tokenizer,
                model=app.state.model,
                messages=messages,
                max_new_tokens=request.generation.max_new_tokens,
                temperature=request.generation.temperature,
                top_p=request.generation.top_p,
                top_k=request.generation.top_k,
                repetition_penalty=request.generation.repetition_penalty,
            ):
                if chunk:
                    yield chunk

        return StreamingResponse(answer_stream(), media_type="text/plain; charset=utf-8")

    # Agent planning endpoints
    @app.post("/agent/plan", response_model=AgentPlanResponse)
    def agent_plan(request: AgentPlanRequest):
        """
        Plan a workflow from a user goal.
        Takes a goal and current context, returns a detailed workflow plan.
        """
        try:
            # Initialize planning engine if not already done
            if not hasattr(app.state, "planning_engine"):
                app.state.planning_engine = PlanningEngine()
            
            engine = app.state.planning_engine
            
            # Convert request to dict for planning
            context = {
                "current_url": request.context.current_url,
                "open_tabs": request.context.open_tabs,
                "available_permissions": request.context.available_permissions,
            }
            
            # Generate workflow plan
            workflow = engine.plan_workflow(request.goal, context)
            
            # Convert to response format
            response_steps = []
            for step in workflow.steps:
                response_steps.append(WorkflowStep(
                    id=step.step_id,
                    type=step.step_type,
                    content=step.content,
                    condition=step.condition,
                ))
            
            return AgentPlanResponse(
                workflow_id=workflow.workflow_id,
                goal=workflow.goal,
                description=workflow.description,
                steps=response_steps,
                required_permissions=workflow.required_permissions,
                risk_assessment=workflow.risk_assessment,
                estimated_duration_seconds=workflow.estimated_duration_seconds,
                created_at=workflow.created_at,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow planning failed: {str(exc)}"
            )

    @app.post("/agent/execute", response_model=AgentExecuteResponse)
    def agent_execute(request: AgentExecuteRequest):
        """
        Execute a single action in a workflow.
        Delegates to browser process but handles response structuring.
        """
        try:
            # In production, this would call the browser process
            # For now, return structured response that browser will populate
            return AgentExecuteResponse(
                step_id=request.step_id,
                action_id=f"{request.workflow_id}_{request.step_id}",
                success=False,  # Will be set by browser execution
                observation={
                    "action_type": request.action_type,
                    "params": request.params,
                    "status": "pending",
                },
                next_step_id=None,  # Will be determined after execution
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Action execution failed: {str(exc)}"
            )

    @app.post("/agent/clarify")
    def agent_clarify(request: ClarificationResponse):
        """
        Handle user response to clarification questions.
        Updates workflow state with user input.
        """
        try:
            # Store clarification response for workflow to continue
            if not hasattr(app.state, "clarifications"):
                app.state.clarifications = {}
            
            app.state.clarifications[request.workflow_id] = {
                "step_id": request.step_id,
                "response": request.response,
                "approval": request.approval,
                "timestamp": time.time(),
            }
            
            return {
                "ok": True,
                "workflow_id": request.workflow_id,
                "step_id": request.step_id,
                "message": "Clarification received, workflow resuming...",
            }
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Clarification handling failed: {str(exc)}"
            )

    @app.post("/agent/observe", response_model=ObservationResponse)
    def agent_observe(request: ObservationRequest):
        """
        Analyze workflow observations and determine if adaptation is needed.
        Returns analysis and suggested adaptations.
        """
        try:
            if not hasattr(app.state, "observation_analyzer"):
                app.state.observation_analyzer = ObservationAnalyzer()
            
            analyzer = app.state.observation_analyzer
            
            # Analyze observations
            all_successful = all(obs.success for obs in request.observations)
            needs_adaptation = not all_successful
            suggested_actions = []
            
            # Analyze each observation
            for obs in request.observations:
                if not obs.success and obs.error:
                    analysis = analyzer.analyze_observation(
                        {
                            "success": obs.success,
                            "error": obs.error,
                        },
                        {"action_type": obs.type}
                    )
                    if analysis["needs_adaptation"]:
                        needs_adaptation = True
                        suggested_actions.extend(analysis["suggested_actions"])
            
            return ObservationResponse(
                workflow_id=request.workflow_id,
                analysis={
                    "observations_count": len(request.observations),
                    "successful_count": sum(1 for obs in request.observations if obs.success),
                    "failed_count": sum(1 for obs in request.observations if not obs.success),
                },
                needs_adaptation=needs_adaptation,
                suggested_actions=list(set(suggested_actions)),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Observation analysis failed: {str(exc)}"
            )

    @app.post("/v1/stt/transcribe", response_model=SttResponse)
    async def stt_transcribe(audio: UploadFile = File(...)):
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="No audio payload was provided.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_bytes)

        try:
            upload_mime = (audio.content_type or "").lower()
            transcription_source: str | np.ndarray = temp_path

            if suffix.lower() == ".wav" or "wav" in upload_mime:
                try:
                    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        sample_rate = wav_file.getframerate()
                        frame_count = wav_file.getnframes()
                        pcm_frames = wav_file.readframes(frame_count)

                    if sample_width != 2:
                        raise ValueError(f"Unsupported WAV bit depth: {sample_width * 8}. Expected 16-bit PCM.")

                    pcm = np.frombuffer(pcm_frames, dtype=np.int16)
                    if channels > 1:
                        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)

                    pcm_f32 = pcm.astype(np.float32) / 32768.0
                    if sample_rate != 16000 and pcm_f32.size:
                        target_count = max(1, int(round(pcm_f32.size * (16000.0 / sample_rate))))
                        src_idx = np.linspace(0, pcm_f32.size - 1, num=pcm_f32.size)
                        dst_idx = np.linspace(0, pcm_f32.size - 1, num=target_count)
                        pcm_f32 = np.interp(dst_idx, src_idx, pcm_f32).astype(np.float32)
                    transcription_source = pcm_f32
                except Exception:
                    transcription_source = temp_path

            def run_transcription(stt_model, source):
                segments, _info = stt_model.transcribe(
                    source,
                    language=get_stt_language(),
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                text_local = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
                if not text_local:
                    segments, _info = stt_model.transcribe(
                        source,
                        language=get_stt_language(),
                        beam_size=1,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                    )
                    text_local = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
                return text_local

            stt_model = get_stt_model(app)
            text = run_transcription(stt_model, transcription_source)
            if not text:
                raise HTTPException(status_code=422, detail="Speech transcription returned empty text.")
            return {"text": text}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("STT transcription failed before CPU fallback")
            strict_gpu = (os.getenv("NEXA_STT_DEVICE", "auto").strip().lower() or "auto") == "cuda"
            if strict_gpu:
                raise HTTPException(
                    status_code=500,
                    detail=f"Speech transcription failed in GPU-only mode: {exc}",
                ) from exc
            runtime = getattr(app.state, "stt_model_runtime", {}) or {}
            if runtime.get("device") != "cpu":
                try:
                    from faster_whisper import WhisperModel

                    model_size = runtime.get("model_size") or os.getenv("NEXA_STT_MODEL", "tiny")
                    app.state.stt_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    app.state.stt_model_runtime = {
                        "model_size": model_size,
                        "device": "cpu",
                        "compute_type": "int8",
                    }
                    text = run_transcription(app.state.stt_model, transcription_source)
                    if text:
                        return {"text": text}
                    raise HTTPException(status_code=422, detail="Speech transcription returned empty text.")
                except HTTPException:
                    raise
                except Exception as retry_exc:
                    logger.exception("STT transcription failed after CPU fallback")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Speech transcription failed: {retry_exc}",
                    ) from retry_exc
            raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}") from exc
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @app.post("/v1/wake/detect", response_model=WakeDetectResponse)
    async def wake_detect(request: WakeDetectRequest):
        threshold = get_wake_threshold()
        consecutive_hits_required = get_wake_consecutive_hits()
        cooldown_seconds = get_wake_cooldown_seconds()

        if request.reset:
            reset_wake_detector(app, request.session_id)
            return {
                "detected": False,
                "score": 0.0,
                "threshold": threshold,
                "supported": is_wake_detection_enabled(),
                "model": os.path.splitext(os.path.basename(get_wake_model_path()))[0] if get_wake_model_path() else None,
            }

        if not is_wake_detection_enabled():
            return {
                "detected": False,
                "score": 0.0,
                "threshold": threshold,
                "supported": False,
                "model": None,
            }

        if request.sample_rate != 16000:
            raise HTTPException(status_code=400, detail="Wake detection expects 16kHz PCM audio.")
        if not request.audio_b64:
            raise HTTPException(status_code=400, detail="Wake detection audio payload is required.")

        try:
            audio_bytes = base64.b64decode(request.audio_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Wake detection audio decode failed: {exc}") from exc

        if not audio_bytes or len(audio_bytes) % 2 != 0:
            raise HTTPException(status_code=400, detail="Wake detection audio payload is invalid.")

        audio_frame = np.frombuffer(audio_bytes, dtype=np.int16)

        try:
            detector = get_wake_detector(app, request.session_id)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=f"Wake detection dependency missing: {exc}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            prediction = detector["model"].predict(audio_frame)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Wake detection failed: {exc}") from exc

        target_model = detector["target_model"]
        raw_score = prediction.get(target_model)
        if raw_score is None and prediction:
            target_model, raw_score = max(prediction.items(), key=lambda item: float(item[1]))

        raw_score_value = float(raw_score or 0.0)
        detector["recent_scores"].append(raw_score_value)
        peak_score = float(max(detector["recent_scores"])) if detector["recent_scores"] else raw_score_value
        score = raw_score_value

        if peak_score >= threshold:
            detector["consecutive_hits"] += 1
        else:
            detector["consecutive_hits"] = 0

        now = time.time()
        cooldown_active = (now - detector["last_detection_at"]) < cooldown_seconds
        detected = (
            not cooldown_active
            and detector["consecutive_hits"] >= consecutive_hits_required
        )
        if detected:
            detector["last_detection_at"] = now
            detector["consecutive_hits"] = 0
            detector["recent_scores"].clear()

        return {
            "detected": detected,
            "score": peak_score,
            "threshold": threshold,
            "supported": True,
            "model": target_model,
        }

    @app.post("/v1/tts/speak")
    async def tts_speak(request: TtsRequest):
        try:
            audio_bytes, mime_type = await synthesize_tts_audio(
                text=request.text.strip(),
                voice=request.voice,
                rate=request.rate,
                voice_id=request.voice_id,
                model_id=request.model_id,
            )
        except ImportError as exc:
            missing = str(exc)
            return Response(
                content=json.dumps({"error": f"Missing dependency: {missing}"}),
                media_type="application/json",
                status_code=503,
            )
        except RuntimeError as exc:
            return Response(
                content=json.dumps({"error": f"TTS configuration error: {exc}"}),
                media_type="application/json",
                status_code=400,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            # Check if it's an API error from ElevenLabs
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                status_code = 401
            elif "429" in error_msg or "rate" in error_msg.lower():
                status_code = 429
            elif "500" in error_msg or "service" in error_msg.lower():
                status_code = 502
            else:
                status_code = 502
            
            return Response(
                content=json.dumps({"error": f"TTS synthesis failed: {error_msg}"}),
                media_type="application/json",
                status_code=status_code,
            )

        return Response(content=audio_bytes, media_type=mime_type)

    return app


def parse_args():
    profile = load_profile()
    parser = argparse.ArgumentParser(description="Serve your local Qwen model with FastAPI.")
    parser.add_argument("--model_name", default=profile.get("model_name", DEFAULT_MODEL))
    parser.add_argument("--adapter_dir", default=os.getenv("QWEN_ADAPTER_DIR"))
    parser.add_argument("--system_prompt", default=profile.get("system_prompt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


profile = load_profile()
app = build_app(
    model_name=os.getenv("QWEN_MODEL_NAME", profile.get("model_name", DEFAULT_MODEL)),
    adapter_dir=os.getenv("QWEN_ADAPTER_DIR"),
    system_prompt=os.getenv("NEXA_SYSTEM_PROMPT", profile.get("system_prompt")),
    assistant_name=profile.get("assistant_name", "Nexa"),
)


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        build_app(
            model_name=args.model_name,
            adapter_dir=args.adapter_dir,
            system_prompt=args.system_prompt,
            assistant_name=profile.get("assistant_name", "Nexa"),
        ),
        host=args.host,
        port=args.port,
    )
