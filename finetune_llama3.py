from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Model and dataset information
model_id = "unsloth/llama-3-8b-bnb-4bit"
dataset_id = "shailja/Verilog_GitHub"
output_dir = "output/llama3_finetune"

# Load dataset
dataset = load_dataset(dataset_id, split="train")

# Load model and tokenizer
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# Prepare model for LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    max_seq_length=max_seq_length
)

# Fine-tuning parameters
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    output_dir=output_dir,
    optim="adamw_8bit",
    seed=3407,
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

# Training
trainer.train()

print("Fine-tuning completed")
