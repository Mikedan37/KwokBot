
# ocr_only_to_jsonl.py — KwokBot fallback extractor

import os
import base64
import json
import time
import fitz  # PyMuPDF
import requests
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict

# Load API keys
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PDF_FOLDERS = [
    os.path.join(ROOT, "materials", "Slides"),
    os.path.join(ROOT, "materials", "TextBook")
]
OUTPUT_IMAGES = os.path.join(ROOT, "output_images")
OUTPUT_JSONL = os.path.join(ROOT, "data", "kwokbot_fallback.jsonl")
os.makedirs(OUTPUT_IMAGES, exist_ok=True)

# Helpers
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        pix = doc.load_page(page_num).get_pixmap(dpi=300)
        img_path = os.path.join(OUTPUT_IMAGES, f"{os.path.basename(pdf_path).replace('.pdf','')}_page_{page_num}.png")
        pix.save(img_path)
        images.append((page_num, img_path))
    return images

def mathpix_image_ocr(image_path):
    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode()
    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_APP_KEY,
        "Content-type": "application/json"
    }
    payload = {
        "src": f"data:image/png;base64,{b64_img}",
        "formats": ["text"]
    }
    res = requests.post("https://api.mathpix.com/v3/text", headers=headers, json=payload)
    return res.json().get("text", "")

def process_pdf(pdf_path, tag_counter):
    entries = []
    print(f"[🧠] OCR-only processing: {os.path.basename(pdf_path)}")
    for page_num, image_path in convert_pdf_to_images(pdf_path):
        text = mathpix_image_ocr(image_path)
        if len(text.strip()) < 10:
            continue
        tags = ["ocr"]
        tag_counter["ocr"] += 1
        entries.append({
            "instruction": "Explain or derive the following expression or concept from EE 140 class:",
            "input": "",
            "output": text.strip(),
            "meta": {
                "source": os.path.basename(pdf_path),
                "page": page_num,
                "tags": tags
            }
        })
    return entries

if __name__ == "__main__":
    all_entries = []
    tag_counter = defaultdict(int)

    pdf_files = []
    for folder in PDF_FOLDERS:
        if os.path.exists(folder):
            pdf_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")])

    with tqdm(total=len(pdf_files), desc="OCR Fallback Progress", unit="pdf") as overall:
        for pdf_path in pdf_files:
            all_entries.extend(process_pdf(pdf_path, tag_counter))
            overall.update(1)

    with open(OUTPUT_JSONL, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")

    print(f"\n[✓] Fallback OCR done! {len(all_entries)} entries saved to {OUTPUT_JSONL}")

