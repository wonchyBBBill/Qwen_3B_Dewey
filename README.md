# Qwen_Dewey

A complete pipeline for building a domain-specific fine-tune of **Qwen 2.5-3B** on John Dewey's *Art as Experience* (1934). This project includes the full data preparation pipeline, LoRA training scripts optimized for Apple Silicon, and an evaluation framework.

## 🚀 Project Overview

This repository provides a structured approach to transforming a source PDF of a philosophical text into a specialized LLM adapter. The process follows these stages:
1. **Extraction**: PDF $\rightarrow$ Raw Text (via `pdfplumber` or Tesseract OCR).
2. **Cleaning**: Removing OCR artifacts, marginal annotations, and page headers.
3. **Structuring**: Dividing the text into 14 distinct chapters.
4. **Preprocessing**: Tokenizing and packing text into JSONL format for SFT.
5. **Training**: Fine-tuning Qwen 2.5-3B using LoRA (via Unsloth on MPS).
6. **Evaluation**: Measuring model performance using custom benchmarks and a judge LLM.

---

## 📂 Repository Structure

### 🛠️ Core Scripts
| Script | Purpose |
| :--- | :--- |
| `ocr_extract.py` | Extracts text from PDF using Tesseract OCR. |
| `clean_annotation.py` | Strips marginal annotations from the text. |
| `remove_header.py` | Removes recurring page headers. |
| `arrange_newlines.py` | Fixes soft-wrapped lines and hyphenation. |
| `divide_by_chapters.py` | Splits the cleaned book into 14 chapter files. |
| `preprocess.py` | Creates `dewey_train.jsonl` and `dewey_test.jsonl`. |
| `train_dewey_3b.py` | LoRA fine-tuning script for Apple Silicon. |
| `perplexity.py` | Measures model perplexity on the test set. |
| `evaluate_dewey.py` | Runs domain-specific evaluations. |
| `evaluate_general.py` | Runs general capability evaluations. |

### 📊 Data & Models
- `Art_as_Experience_Chapters/`: The finalized text split by chapter.
- `data/`: Training and testing JSONL files.
- `dewey_lora/`: The trained LoRA adapters and tokenizer configurations.
- `evaluation/`: Contains `questions.json` and the `judge.py` script for automated evaluation.

---

## ⚙️ Setup & Usage

### 1. Dependencies
Install OS-level dependencies:
```bash
brew install tesseract poppler
```
Install Python libraries:
```bash
pip install pdfplumber pytesseract pdf2image transformers datasets trl torch unsloth
```

### 2. Execution Pipeline
Run the scripts in the following order to reproduce the model:
```bash
# Extraction & Cleaning
python ocr_extract.py
python clean_annotation.py
python remove_header.py
python arrange_newlines.py

# Structuring & Preprocessing
python divide_by_chapters.py
python preprocess.py

# Training
python train_dewey_3b.py
```

### 3. Evaluation
To evaluate the fine-tuned model against the base model:
```bash
python evaluate_dewey.py
python evaluate_general.py
```

---

## 🧠 Training Configuration

- **Base Model**: `Qwen/Qwen2.5-3B`
- **Method**: LoRA (Rank 16, Alpha 16)
- **Quantization**: 4-bit
- **Hardware**: Apple Silicon (`mps`)
- **Sequence Length**: 2048
- **Epochs**: 2
- **Learning Rate**: 2e-4

---

## 📖 Source Material
The corpus is **John Dewey, *Art as Experience* (1934)**. The text consists of 14 chapters covering the nature of experience, the act of expression, and the organization of energies in art.

---

## ⚖️ License
- **Source Text**: Public Domain.
- **Code**: MIT License.
