from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import os

# ========== CONFIG ==========
model_path = "/Users/mdanylchuk/Documents/FKwokBot/models/mistral-7b-hf"
data_path = "/Users/mdanylchuk/Documents/FKwokBot/data/kwokbot_train.jsonl"
output_dir = "/Users/mdanylchuk/Documents/FKwokBot/output/kwokbot-lora"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"💻 Using device: {device}")

# ========== LOAD TOKENIZER ==========
print("🔓 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== LOAD BASE MODEL ==========
print("🧠 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# ========== APPLY LoRA ==========
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config).to(device)
print("🛠️ LoRA applied!")

# ========== LOAD & FORMAT DATA ==========
print("📚 Loading dataset...")
dataset = load_dataset("json", data_files=data_path)["train"]
split = dataset.train_test_split(test_size=0.1)
train_dataset = split["train"]
eval_dataset = split["test"]

def format_prompt(entry):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{entry['instruction']}

### Response:
{entry['output']}"""

def tokenize(entry):
    return tokenizer(
        format_prompt(entry),
        truncation=True,
        max_length=512,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize)
eval_dataset = eval_dataset.map(tokenize)

# ========== TRAINING SETUP ==========
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",  # ❌ Disable eval for now
    fp16=False,
    bf16=False,
    push_to_hub=False,
    report_to="tensorboard",
    logging_dir=os.path.join(output_dir, "logs")
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # ❌ Disable eval for now
    tokenizer=None,
    data_collator=data_collator
)

# ========== TRAIN ==========
print("🚀 Starting fine-tuning...")
trainer.train()

print("💾 Saving...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
