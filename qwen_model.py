import os
import re
from threading import Thread
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TextIteratorStreamer


DEFAULT_MODEL = str(Path(__file__).resolve().parent / "models" / "Qwen3-4B-Instruct-2507")
MOJIBAKE_REPLACEMENTS = {
    "Ã¢â‚¬â„¢": "'",
    "Ã¢â‚¬Å“": '"',
    "Ã¢â‚¬Â": '"',
    "Ã¢â‚¬â€œ": "-",
    "Ã¢â‚¬â€": "-",
    "Ã¢â‚¬Â¦": "...",
    "Ã°Å¸â€˜â€¹": "ðŸ‘‹",
}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_model(model_name: str, adapter_dir: str | None):
    offload_dir = os.getenv("NEXA_OFFLOAD_DIR", os.path.join(os.getcwd(), "model_offload"))
    os.makedirs(offload_dir, exist_ok=True)
    force_cuda_only = env_flag("NEXA_FORCE_CUDA_ONLY", default=False)
    raw_load_in_4bit = os.getenv("NEXA_LOAD_IN_4BIT")
    load_in_4bit = (
        raw_load_in_4bit.strip().lower() in {"1", "true", "yes", "on"}
        if raw_load_in_4bit is not None
        else torch.cuda.is_available() and not force_cuda_only
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"low_cpu_mem_usage": True}

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "NEXA_LOAD_IN_4BIT=1 was requested, but bitsandbytes is not installed."
            ) from exc

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16 if torch.cuda.is_available() else torch.float32
            ),
        )
        load_kwargs["device_map"] = "auto"
    elif force_cuda_only:
        if not torch.cuda.is_available():
            raise RuntimeError("NEXA_FORCE_CUDA_ONLY=1 was requested, but CUDA is not available.")
        load_kwargs["torch_dtype"] = "auto"
    else:
        load_kwargs.update(
            {
                "torch_dtype": torch.float32,
                "device_map": "cpu",
            }
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if force_cuda_only and not load_in_4bit:
        model = model.to("cuda")

    if adapter_dir:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError(
                "Failed to load adapter because PEFT could not be imported. "
                "Install a PEFT/Transformers compatible stack (for example, pin transformers<5)."
            ) from exc

        adapter_kwargs = {"low_cpu_mem_usage": True}
        if not force_cuda_only:
            adapter_kwargs["offload_dir"] = offload_dir
        model = PeftModel.from_pretrained(model, adapter_dir, **adapter_kwargs)

    model.eval()
    return tokenizer, model


def normalize_text_artifacts(text: str) -> str:
    for source, target in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(source, target)
    return text


def clean_model_output(text: str) -> str:
    text = normalize_text_artifacts(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    text = text.replace(" ;", ";")
    text = text.replace(" ?", "?")
    text = text.replace(" !", "!")
    return text.strip()


def clean_stream_chunk(text: str) -> str:
    text = normalize_text_artifacts(text)
    text = text.replace("\r", "")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" :", ":")
    return text


def get_generation_device(model) -> torch.device:
    input_embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    if input_embeddings is not None:
        weight = getattr(input_embeddings, "weight", None)
        if weight is not None and weight.device.type != "meta":
            return weight.device

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        preferred_prefixes = ("cuda", "xpu", "mps", "npu")
        for prefix in preferred_prefixes:
            for location in hf_device_map.values():
                if isinstance(location, str) and location.startswith(prefix):
                    return torch.device(location)

        for location in hf_device_map.values():
            if isinstance(location, str) and location == "cpu":
                return torch.device("cpu")

    try:
        first_param_device = next(model.parameters()).device
        if first_param_device.type != "meta":
            return first_param_device
    except StopIteration:
        pass

    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device) and model_device.type != "meta":
        return model_device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_reply(
    tokenizer,
    model,
    messages,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    generation_device = get_generation_device(model)
    inputs = {key: value.to(generation_device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return clean_model_output(tokenizer.decode(new_tokens, skip_special_tokens=True))


def stream_reply(
    tokenizer,
    model,
    messages,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    generation_device = get_generation_device(model)
    inputs = {key: value.to(generation_device) for key, value in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "top_k": top_k,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": 3,
        "streamer": streamer,
    }

    generation_error: list[BaseException] = []

    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        except BaseException as exc:  # noqa: BLE001
            generation_error.append(exc)
        finally:
            streamer.end()

    thread = Thread(target=run_generation, daemon=True)
    thread.start()

    for text in streamer:
        if text:
            yield clean_stream_chunk(text)

    thread.join()
    if generation_error:
        raise generation_error[0]
