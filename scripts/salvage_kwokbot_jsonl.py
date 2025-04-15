# salvage_kwokbot_jsonl.py – Attempt to recover valuable entries from broken JSON fragments

import os
import json
from tqdm import tqdm

INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_train.jsonl"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_salvaged.jsonl"))

buffer = []
salvaged = 0

with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for line in tqdm(infile, desc="Scanning for salvageable entries"):
        stripped = line.strip()
        if not stripped:
            continue

        buffer.append(stripped)

        # Try to join the buffer and parse it as JSON
        try:
            candidate = json.loads("".join(buffer))
            if all(k in candidate for k in ["instruction", "output"]):
                outfile.write(json.dumps(candidate) + "\n")
                salvaged += 1
            buffer = []  # clear the buffer
        except json.JSONDecodeError:
            continue  # wait for more lines to complete the object

print(f"[✓] Salvaged {salvaged} entries and saved to: {OUTPUT_FILE}")

