# evaluate_dewey.py

import json
from pathlib import Path

import torch
from unsloth import FastLanguageModel


QUESTIONS = Path("evaluation/questions.json")

BASE_MODEL = "Qwen/Qwen2.5-3B"
LORA_MODEL = "./dewey_lora"

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 800


def load_questions():
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def load_model(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def answer_question(model, tokenizer, question):

    prompt = f"""Answer the following philosophical question carefully.

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,

            # deterministic decoding
            do_sample=False,

            temperature=None,
            top_p=None,
        )

    # Only decode newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return answer.strip()


def run(model_path, output_path):

    print(f"\nLoading: {model_path}")

    model, tokenizer = load_model(model_path)

    questions = load_questions()

    results = []

    for i, q in enumerate(questions):

        print(
            f"[{i + 1}/{len(questions)}] "
            f"{q['id']}"
        )

        answer = answer_question(
            model,
            tokenizer,
            q["question"],
        )

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "rubric": q["rubric"],
            "answer": answer,
        })

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":

    run(
        BASE_MODEL,
        "evaluation/answers_base.json",
    )

    run(
        LORA_MODEL,
        "evaluation/answers_dewey.json",
    )