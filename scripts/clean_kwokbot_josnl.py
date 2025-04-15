# clean_kwokbot_jsonl.py – Removes invalid entries and prepares Alpaca-style JSONL for Axolotl fine-tuning

import os
import json
from tqdm import tqdm

INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_train.jsonl"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_train_clean.jsonl"))

valid = 0
invalid = 0

with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for line in tqdm(infile, desc="Cleaning KwokBot JSONL"):
        try:
            entry = json.loads(line)

            # Check Alpaca format structure
            if not isinstance(entry, dict):
                invalid += 1
                continue

            if "instruction" not in entry or "output" not in entry:
                invalid += 1
                continue

            if not entry["instruction"].strip() or not entry["output"].strip():
                invalid += 1
                continue

            # Optional: strip unnecessary whitespace
            entry["instruction"] = entry["instruction"].strip()
            entry["input"] = entry.get("input", "").strip()
            entry["output"] = entry["output"].strip()

            outfile.write(json.dumps(entry) + "\n")
            valid += 1
        except json.JSONDecodeError:
            invalid += 1

print(f"[✓] Cleaned! {valid} valid entries saved to: {OUTPUT_FILE}")
print(f"[!] {invalid} invalid entries were removed.")

