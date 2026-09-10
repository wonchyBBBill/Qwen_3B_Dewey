# train_dewey.py
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-3B"

MAX_SEQ_LENGTH = 2048


# --------------------------------------------------
# Load model
# --------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map="mps",
)


# --------------------------------------------------
# Add LoRA
# --------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,

    r=16,

    lora_alpha=16,

    lora_dropout=0,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=42,
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = load_dataset(
    "json",
    data_files="data/dewey_train.jsonl",
    split="train",
)


# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = SFTTrainer(

    model=model,

    tokenizer=tokenizer,

    train_dataset=dataset,

    dataset_text_field="text",

    max_seq_length=MAX_SEQ_LENGTH,

    packing=True,

    args=SFTConfig(

        output_dir="./dewey_lora",

        num_train_epochs=2,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        logging_steps=10,

        save_strategy="epoch",

        fp16=True,

        seed=42,
    ),
)


# --------------------------------------------------
# Train
# --------------------------------------------------

trainer.train()


# --------------------------------------------------
# Save LoRA adapter
# --------------------------------------------------

model.save_pretrained("./dewey_lora")
tokenizer.save_pretrained("./dewey_lora")