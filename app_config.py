import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "data" / "nexa_profile.json"
DEFAULT_SYSTEM_PROMPT = "You are Nexa, a helpful assistant developed by Nexa Labs and built on Qwen3."
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen3-4B-Instruct-2507")


def load_profile(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {
            "assistant_name": "Nexa",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "model_name": DEFAULT_MODEL_PATH,
        }

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "assistant_name": data.get("assistant_name", "Nexa"),
        "system_prompt": data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        "model_name": data.get("model_name", DEFAULT_MODEL_PATH),
    }
