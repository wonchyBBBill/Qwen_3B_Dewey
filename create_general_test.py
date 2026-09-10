# create_general_test.py

import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42
N_MMLU = 100
N_GSM8K = 100

OUTPUT = Path("evaluation/general_test.json")

random.seed(SEED)


# =========================================================
# MMLU
# =========================================================

print("Loading MMLU...")

mmlu = load_dataset(
    "cais/mmlu",
    "all",
    split="test"
)

# Group by subject
by_subject = {}

for item in mmlu:
    subject = item["subject"]
    by_subject.setdefault(subject, []).append(item)


subjects = sorted(by_subject.keys())

print(f"MMLU subjects: {len(subjects)}")


# Sample approximately evenly across subjects
mmlu_questions = []

for i in range(N_MMLU):

    subject = subjects[i % len(subjects)]

    item = random.choice(by_subject[subject])

    mmlu_questions.append({
        "id": f"MMLU_{i+1:03d}",
        "type": "mmlu",
        "subject": subject,
        "question": item["question"],
        "choices": item["choices"],
        "answer": item["answer"]
    })


# =========================================================
# GSM8K
# =========================================================

print("Loading GSM8K...")

gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split="test"
)

indices = random.sample(
    range(len(gsm8k)),
    N_GSM8K
)

gsm_questions = []

for i, idx in enumerate(indices):

    item = gsm8k[idx]

    gsm_questions.append({
        "id": f"GSM8K_{i+1:03d}",
        "type": "gsm8k",
        "question": item["question"],
        "answer": item["answer"]
    })


# =========================================================
# Save
# =========================================================

questions = (
    mmlu_questions +
    gsm_questions
)

output = {
    "benchmark": "general_capability_regression_v1",
    "seed": SEED,
    "mmlu_n": N_MMLU,
    "gsm8k_n": N_GSM8K,
    "questions": questions
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
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
print(f"Created {len(questions)} questions.")
print(f"MMLU:  {N_MMLU}")
print(f"GSM8K: {N_GSM8K}")
print(f"Saved to: {OUTPUT}")