import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


_original_read_text = Path.read_text


def _utf8_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = "utf-8"
    return _original_read_text(self, encoding=encoding, errors=errors)


Path.read_text = _utf8_read_text
from trl import SFTConfig, SFTTrainer
Path.read_text = _original_read_text


DEFAULT_MODEL = "models/Qwen3-4B-Instruct-2507"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-4B-Instruct-2507 with LoRA.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset_path", required=True, help="Path to a JSONL dataset.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--assistant_only_loss", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def format_example(tokenizer, example: dict[str, Any]) -> dict[str, str]:
    if example.get("messages"):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    if "prompt" in example and "response" in example:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    raise ValueError("Each dataset row must contain either `messages` or `prompt` + `response`.")


def normalize_conversation(example: dict[str, Any]) -> dict[str, Any]:
    if example.get("messages"):
        return {"messages": example["messages"]}

    if "prompt" in example and "response" in example:
        return {
            "messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        }

    raise ValueError("Each dataset row must contain either `messages` or `prompt` + `response`.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    if args.assistant_only_loss:
        train_dataset = raw_dataset.map(
            normalize_conversation,
            remove_columns=raw_dataset.column_names,
        )
        dataset_text_field = None
    else:
        train_dataset = raw_dataset.map(
            lambda row: format_example(tokenizer, row),
            remove_columns=raw_dataset.column_names,
        )
        dataset_text_field = "text"

    peft_config = None
    if not args.full_finetune:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_length,
        assistant_only_loss=args.assistant_only_loss,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }
    if dataset_text_field:
        trainer_kwargs["dataset_text_field"] = dataset_text_field

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved training output to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
