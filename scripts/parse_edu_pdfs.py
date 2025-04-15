import os
import json
from pathlib import Path
from datetime import datetime
from pix2text import Pix2Text

# Paths
INPUT_DIR = Path("~/Documents/FKwokBot/output_images").expanduser()
OUTPUT_PATH = Path("~/Documents/FKwokBot/data/kwokbot_textbook_pix2text.jsonl").expanduser()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Pix2Text Init
p2t = Pix2Text()

# Helper to get chapter number
def guess_chapter_from_filename(name: str):
    import re
    match = re.search(r"Chapter[_\s]*(\d+)", name)
    return int(match.group(1)) if match else None

# Main Parsing Loop
entries = []
for img_path in sorted(INPUT_DIR.glob("*.png")):
    print(f"🔍 Scanning {img_path.name}...")
    try:
        result = p2t(img_path)

        # Normalize result
        if isinstance(result, dict):
            result = [result]

        chapter = guess_chapter_from_filename(img_path.name)

        for i, chunk in enumerate(result):
            text = chunk.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            entries.append({
                "instruction": "Use this textbook image chunk for reference.",
                "input": "",
                "output": text,
                "meta": {
                    "source": str(img_path),
                    "filename": img_path.name,
                    "chunk_id": i,
                    "type": "TextBook",
                    "timestamp": datetime.now().isoformat(),
                    "timestamp_cleaned": datetime.now().isoformat(),
                    "cleaned": True,
                    "chapter": chapter
                }
            })
    except Exception as e:
        print(f"❌ Failed on {img_path.name}: {e}")

# Save
with open(OUTPUT_PATH, "w") as f:
    for entry in entries:
        json.dump(entry, f)
        f.write("\n")

print(f"\n✅ DONE — Parsed {len(entries)} chunks from PNGs in {INPUT_DIR}")
print(f"📄 Saved to {OUTPUT_PATH}")