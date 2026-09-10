# evaluate_general.py

import json
import re
from pathlib import Path

import torch
from unsloth import FastLanguageModel


QUESTIONS_FILE = "evaluation/general_test.json"

BASE_OUTPUT = "evaluation/general_base_results.json"
DEWEY_OUTPUT = "evaluation/general_dewey_results.json"


# =========================================================
# MODEL LOADING
# =========================================================

def load_base_model():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B",
        max_seq_length=4096,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def load_dewey_model():

    # Same base model loading as above,
    # then attach the LoRA adapter.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B",
        max_seq_length=4096,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model
    )

    # If your Unsloth version loads the saved adapter
    # differently, use the exact adapter-loading code
    # from your existing Dewey evaluation script here.

    FastLanguageModel.for_inference(model)

    return model, tokenizer

# =========================================================
# LOAD QUESTIONS
# =========================================================

with open(
    QUESTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)

questions = benchmark["questions"]


# =========================================================
# GENERATION
# =========================================================

def generate_answer(
    model,
    tokenizer,
    question
):

    if question["type"] == "mmlu":

        prompt = f"""Answer the following multiple-choice question.

Return only the letter of the correct answer: A, B, C, or D.

Question:
{question["question"]}

A. {question["choices"][0]}
B. {question["choices"][1]}
C. {question["choices"][2]}
D. {question["choices"][3]}

Answer:"""

    else:

        prompt = f"""Solve the following math problem.

Show your reasoning and give the final numerical answer.

Question:
{question["question"]}

Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    generated = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# =========================================================
# MMLU PARSING
# =========================================================

def extract_mmlu_answer(text):

    text = text.upper().strip()

    # Look for a standalone A/B/C/D.
    matches = re.findall(
        r"\b([ABCD])\b",
        text
    )

    if matches:
        return matches[-1]

    return None


# =========================================================
# GSM8K PARSING
# =========================================================

def extract_number(text):

    # Remove commas and dollar signs.
    text = text.replace(",", "")
    text = text.replace("$", "")

    # Prefer a number after "answer" or "####".
    matches = re.findall(
        r"(?:####|answer\s*(?:is|:)?\s*)"
        r"(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1]

    # Otherwise take the last number.
    matches = re.findall(
        r"-?\d+(?:\.\d+)?",
        text
    )

    if matches:
        return matches[-1]

    return None


def extract_gsm8k_gold(text):

    match = re.search(
        r"####\s*(-?\d+(?:\.\d+)?)",
        text
    )

    if match:
        return match.group(1)

    return None


# =========================================================
# RUN BENCHMARK
# =========================================================

def evaluate(
    model,
    tokenizer,
    output_file,
    model_name
):

    results = []

    mmlu_correct = 0
    mmlu_total = 0

    gsm_correct = 0
    gsm_total = 0

    for i, q in enumerate(questions):

        print(
            f"{model_name}: "
            f"{i + 1}/{len(questions)} "
            f"{q['id']}"
        )

        answer = generate_answer(
            model,
            tokenizer,
            q
        )

        result = {
            "id": q["id"],
            "type": q["type"],
            "subject": q.get("subject"),
            "question": q["question"],
            "model_answer": answer
        }

        # -----------------------------------------------
        # MMLU
        # -----------------------------------------------

        if q["type"] == "mmlu":

            predicted = extract_mmlu_answer(
                answer
            )

            gold = (
                ["A", "B", "C", "D"][
                    q["answer"]
                ]
            )

            correct = predicted == gold

            result["predicted"] = predicted
            result["correct_answer"] = gold
            result["correct"] = correct

            mmlu_total += 1

            if correct:
                mmlu_correct += 1

        # -----------------------------------------------
        # GSM8K
        # -----------------------------------------------

        elif q["type"] == "gsm8k":

            predicted = extract_number(
                answer
            )

            gold = extract_gsm8k_gold(
                q["answer"]
            )

            correct = predicted == gold

            result["predicted"] = predicted
            result["correct_answer"] = gold
            result["correct"] = correct

            gsm_total += 1

            if correct:
                gsm_correct += 1

        results.append(result)

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = {
        "model": model_name,

        "mmlu": {
            "correct": mmlu_correct,
            "total": mmlu_total,
            "accuracy": (
                mmlu_correct / mmlu_total
            )
        },

        "gsm8k": {
            "correct": gsm_correct,
            "total": gsm_total,
            "accuracy": (
                gsm_correct / gsm_total
            )
        },

        "overall": {
            "correct":
                mmlu_correct + gsm_correct,

            "total":
                mmlu_total + gsm_total,

            "accuracy":
                (
                    mmlu_correct + gsm_correct
                )
                /
                (
                    mmlu_total + gsm_total
                )
        }
    }

    output = {
        "benchmark": benchmark,
        "model": model_name,
        "summary": summary,
        "results": results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    print(
        f"MMLU:  "
        f"{mmlu_correct}/{mmlu_total} "
        f"({mmlu_correct / mmlu_total:.1%})"
    )

    print(
        f"GSM8K: "
        f"{gsm_correct}/{gsm_total} "
        f"({gsm_correct / gsm_total:.1%})"
    )

    print(
        f"Overall: "
        f"{mmlu_correct + gsm_correct}/"
        f"{mmlu_total + gsm_total} "
        f"({(mmlu_correct + gsm_correct) / (mmlu_total + gsm_total):.1%})"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # Base model
    model, tokenizer = load_base_model()

    evaluate(
        model,
        tokenizer,
        BASE_OUTPUT,
        "Qwen2.5-3B-base"
    )

    del model
    torch.cuda.empty_cache()

    # Dewey LoRA
    model, tokenizer = load_dewey_model()

    evaluate(
        model,
        tokenizer,
        DEWEY_OUTPUT,
        "Qwen2.5-3B-Dewey-LoRA"
    )