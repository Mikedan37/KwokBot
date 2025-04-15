# salvage_kwokbot_jsonl.py – Fixes broken JSON entries from a KwokBot training dump

import os
import json
from tqdm import tqdm

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_train.jsonl"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_salvaged.jsonl"))

recovered_entries = []
buffer = []
valid = 0

with open(INPUT_PATH, "r") as infile:
    for line in infile:
        stripped = line.strip()
        if not stripped:
            continue

        buffer.append(stripped)
        try:
            joined = "".join(buffer)
            obj = json.loads(joined)
            # Check for essential keys
            if all(k in obj for k in ["instruction", "output"]):
                recovered_entries.append(obj)
                valid += 1
            buffer = []  # clear buffer on success
        except json.JSONDecodeError:
            # Keep buffering lines
            continue

# Save recovered entries
with open(OUTPUT_PATH, "w") as out:
    for entry in recovered_entries:
        out.write(json.dumps(entry) + "\n")

print(f"[✓] Salvaged {valid} broken-but-valuable entries into: {OUTPUT_PATH}")

