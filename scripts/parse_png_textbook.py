import os
import json
from pathlib import Path
from datetime import datetime
from pix2text import Pix2Text
from collections.abc import Iterable

INPUT_DIR = Path("~/Documents/FKwokBot/output_images").expanduser()
OUTPUT_PATH = Path("~/Documents/FKwokBot/data/kwokbot_textbook_pix2text.jsonl").expanduser()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

p2t = Pix2Text()

ALLOWED_TYPES = {"title", "plain text", "text", "isolate_formula", "formula", "caption", "embedding"}

def guess_chapter_from_filename(name: str):
    import re
    match = re.search(r"Chapter[_\s]*(\d+)", name)
    return int(match.group(1)) if match else None

def ensure_list(x):
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes, dict)):
        return x
    else:
        return [x]

all_entries = []

for img_path in sorted(INPUT_DIR.glob("*.png")):
    print(f"🔍 Scanning {img_path.name}...")
    try:
        result = p2t(img_path)
        chunks = ensure_list(result)

        page_text = []
        for chunk in chunks:
            text = ""
            label = ""

            if isinstance(chunk, dict):
                text = chunk.get("text", "").strip()
                label = chunk.get("type", "").lower()
            elif hasattr(chunk, "text"):
                text = str(chunk.text).strip()
                label = getattr(chunk, "type", "").lower()
            else:
                continue

            if text and len(text) > 10 and label in ALLOWED_TYPES:
                page_text.append(text)

        if not page_text:
            print(f"⚠️  No usable content in {img_path.name}")
            continue

        chapter = guess_chapter_from_filename(img_path.name)
        full_text = "\n".join(page_text)

        all_entries.append({
            "instruction": "Use this textbook image chunk for reference.",
            "input": "",
            "output": full_text,
            "meta": {
                "source": str(img_path),
                "filename": img_path.name,
                "chunk_id": 0,
                "type": "TextBook",
                "timestamp": datetime.now().isoformat(),
                "timestamp_cleaned": datetime.now().isoformat(),
                "cleaned": True,
                "chapter": chapter
            }
        })

    except Exception as e:
        print(f"❌ Failed on {img_path.name}: {e}")

with open(OUTPUT_PATH, "w") as f:
    for entry in all_entries:
        json.dump(entry, f)
        f.write("\n")

print(f"\n✅ DONE — {len(all_entries)} textbook image chunks saved to:")
print(f"📄 {OUTPUT_PATH}")

