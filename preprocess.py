# preprocess.py

from pathlib import Path
import json

from transformers import AutoTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-3B"

CHAPTER_DIR = Path("./Art_as_Experience_Chapters")

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_CHAPTER = "02_CHAPTER_II.txt"

MAX_LENGTH = 2048


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --------------------------------------------------
# Read paragraphs
# --------------------------------------------------

def read_paragraphs(path):
    text = path.read_text(encoding="utf-8")

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Paragraphs are separated by blank lines
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    return paragraphs


# --------------------------------------------------
# Pack paragraphs into token-length chunks
# --------------------------------------------------

def make_chunks(paragraphs, max_length):
    chunks = []
    current = []

    for paragraph in paragraphs:

        # Tokenize this paragraph without adding special tokens
        paragraph_tokens = tokenizer.encode(
            paragraph,
            add_special_tokens=False
        )

        # If one paragraph itself is too long,
        # split it directly.
        if len(paragraph_tokens) > max_length:

            # Flush current chunk
            if current:
                chunks.append("\n".join(current))
                current = []

            for i in range(0, len(paragraph_tokens), max_length):

                piece_tokens = paragraph_tokens[i:i + max_length]

                piece = tokenizer.decode(
                    piece_tokens,
                    skip_special_tokens=True
                )

                chunks.append(piece)

            continue

        # How long would the new chunk be?
        candidate = "\n".join(current + [paragraph])

        candidate_tokens = tokenizer.encode(
            candidate,
            add_special_tokens=False
        )

        if len(candidate_tokens) <= max_length:
            current.append(paragraph)

        else:
            # Save current chunk
            if current:
                chunks.append("\n".join(current))

            # Start a new chunk
            current = [paragraph]

    # Final chunk
    if current:
        chunks.append("\n\n".join(current))

    return chunks


# --------------------------------------------------
# Process training chapters
# --------------------------------------------------

train_chunks = []

chapter_files = sorted(CHAPTER_DIR.glob("*.txt"))

for path in chapter_files:

    if path.name == TEST_CHAPTER:
        continue

    paragraphs = read_paragraphs(path)

    chunks = make_chunks(
        paragraphs,
        MAX_LENGTH
    )

    for chunk in chunks:
        train_chunks.append({
            "text": chunk,
            "chapter": path.stem
        })


# --------------------------------------------------
# Process test chapter separately
# --------------------------------------------------

test_path = CHAPTER_DIR / TEST_CHAPTER

test_paragraphs = read_paragraphs(test_path)

test_chunks_raw = make_chunks(
    test_paragraphs,
    MAX_LENGTH
)

test_chunks = [
    {
        "text": chunk,
        "chapter": test_path.stem
    }
    for chunk in test_chunks_raw
]


# --------------------------------------------------
# Save JSONL
# --------------------------------------------------

def save_jsonl(data, path):

    with path.open("w", encoding="utf-8") as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )


save_jsonl(
    train_chunks,
    OUTPUT_DIR / "dewey_train.jsonl"
)

save_jsonl(
    test_chunks,
    OUTPUT_DIR / "dewey_test.jsonl"
)


# --------------------------------------------------
# Print statistics
# --------------------------------------------------

def count_tokens(data):

    total = 0

    for item in data:

        total += len(
            tokenizer.encode(
                item["text"],
                add_special_tokens=False
            )
        )

    return total


print("Training chunks:", len(train_chunks))
print("Training tokens:", count_tokens(train_chunks))

print("Test chunks:", len(test_chunks))
print("Test tokens:", count_tokens(test_chunks))