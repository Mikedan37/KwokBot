from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
import json

# === CONFIG ===
model_path = "output/kwokbot-lora"
eval_file = "data/kwokbot_eval.jsonl"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === LOAD TOKENIZER AND MODEL ===
print("🔓 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

# === LOAD EVAL QUESTIONS ===
print("📚 Loading eval questions...")
dataset = load_dataset("json", data_files=eval_file)["train"]

# === LETTER GRADE HELPER ===
def letter_grade(percent):
    if percent >= 90: return "A"
    elif percent >= 80: return "B"
    elif percent >= 70: return "C"
    elif percent >= 60: return "D"
    else: return "F"

# === EVALUATE ===
print("🧠 Running evaluation using greedy forward pass...\n")
correct = 0
results = []

for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
    prompt = f"{item['instruction']}\n\n{item['input']}".strip()
    expected = item["output"].strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id  # avoid warning
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    match = expected.lower() in decoded.lower()

    results.append({
        "question": prompt,
        "expected": expected,
        "got": decoded,
        "correct": match
    })

    print(f"--- Question {i+1} ---")
    print(f"✅ Expected: {expected}")
    print(f"🤖 Got: {decoded}")
    print(f"{'✔️ MATCH' if match else '❌ MISMATCH'}\n")

    if match:
        correct += 1

# === FINAL GRADE ===
accuracy = (correct / len(dataset)) * 100
grade = letter_grade(accuracy)
print(f"📊 Final Grade: {correct}/{len(dataset)} correct — {accuracy:.2f}% ({grade})")

# === SAVE RESULTS ===
with open("eval_results_greedy.json", "w") as f:
    json.dump(results, f, indent=2)
