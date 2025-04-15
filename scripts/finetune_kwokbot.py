# kwokbot_finetune.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# MPS check
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load tokenizer + model
model_name = "~/Documents/FKwokBot/models/mistral-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device})

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Load Alpaca-style JSONL
print("Loading dataset...")
dataset = load_dataset("json", data_files="/data/kwokbot_train.jsonl")

# Tokenize
print("Tokenizing...")
def tokenize(sample):
    prompt = sample["instruction"] + "\n" + sample["input"] + "\n"
    response = sample["output"]
    full_text = prompt + response
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

dataset = dataset["train"].map(tokenize)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
training_args = TrainingArguments(
    output_dir="/Users/mdanylchuk/Documents/FKwokBot/models/kwokbot-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="logs",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=False,  # MPS doesn't support fp16
    bf16=False,
    dataloader_num_workers=2,
    save_total_limit=2,
    report_to=["wandb"],
    run_name="kwokbot-mistral-7b-mac"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("/Users/mdanylchuk/Documents/FKwokBot/models/kwokbot-finetuned")

