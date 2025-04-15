from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Paths ===
BASE_MODEL_PATH = "/Users/mdanylchuk/Documents/FKwokBot/models/mistral-7b-hf"
LORA_PATH = "/Users/mdanylchuk/Documents/FKwokBot/output/kwokbot-lora"

# === Load tokenizer and base model ===
print("🔓 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Patch pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🧠 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    device_map={"": "mps"},
)

print("✨ Applying LoRA fine-tuning...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to("mps")
model.eval()

# === Chat loop ===
print("\n🤖 KwokBot Terminal Chat\nType 'exit' to dip out.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Tokenize input with attention mask
    inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=True).to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean response: slice off user prompt if echoed
    if response.startswith(user_input):
        response = response[len(user_input):].strip()

    print(f"KwokBot: {response}\n")
