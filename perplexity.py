# perplexity.py

import math

import mlx.core as mx
import mlx.nn as nn
import torch
from unsloth import FastLanguageModel


BASE_MODEL = "Qwen/Qwen2.5-3B"
LORA_MODEL = "./dewey_lora"

TEST_FILE = "Art_as_Experience_Chapters/02_CHAPTER_II.txt"

MAX_LENGTH = 4096


def load_model(path):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_LENGTH,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer

def is_mlx_model(model) -> bool:
    # Unsloth MLX path / plain mlx-lm models
    return (
        hasattr(model, "parameters") and callable(getattr(model, "parameters", None))
        and not isinstance(model, torch.nn.Module)
    ) or type(model).__module__.startswith("mlx")


def perplexity(model, tokenizer, text):
    # ---- tokenize ----
    # Prefer plain encode to stay backend-agnostic
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        # fallback
        encodings = tokenizer(text, return_tensors=None, add_special_tokens=True)
        ids = encodings["input_ids"] if isinstance(encodings, dict) else encodings

    if is_mlx_model(model):
        # ========== MLX path ==========
        tokens = mx.array(ids)[None, :]  # (1, seq)

        total_loss = 0.0
        total_tokens = 0

        for start in range(0, tokens.shape[1], MAX_LENGTH):
            end = min(start + MAX_LENGTH, tokens.shape[1])
            chunk = tokens[:, start:end]

            if chunk.shape[1] < 2:
                continue

            # logits predict the *next* token
            logits = model(chunk[:, :-1]).astype(mx.float32)   # (1, L-1, vocab)
            targets = chunk[:, 1:]

            # per-token CE
            losses = nn.losses.cross_entropy(logits, targets, reduction="none")  # (1, L-1)
            mx.eval(losses)

            n_tokens = int(losses.size)
            total_loss += float(losses.sum().item())
            total_tokens += n_tokens

        mean_loss = total_loss / total_tokens
        return math.exp(mean_loss)

    else:
        # ========== Original PyTorch / CUDA path ==========
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"]

        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        total_loss = 0.0
        total_tokens = 0

        for start in range(0, input_ids.shape[1], MAX_LENGTH):
            end = min(start + MAX_LENGTH, input_ids.shape[1])
            chunk = input_ids[:, start:end]

            if chunk.shape[1] < 2:
                continue

            with torch.no_grad():
                outputs = model(input_ids=chunk, labels=chunk)

            n_tokens = chunk.shape[1] - 1
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

        mean_loss = total_loss / total_tokens
        return math.exp(mean_loss)


with open(
    TEST_FILE,
    "r",
    encoding="utf-8"
) as f:

    test_text = f.read()


for name, path in [
    ("BASE", BASE_MODEL),
    ("DEWEY_LORA", LORA_MODEL),
]:

    print(f"\n{name}")

    model, tokenizer = load_model(path)

    ppl = perplexity(
        model,
        tokenizer,
        test_text,
    )

    print(
        f"Chapter II perplexity: {ppl:.2f}"
    )