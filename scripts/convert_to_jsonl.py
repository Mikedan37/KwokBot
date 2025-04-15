# kwokbot_converter.py - Unified script to process PDFs into Alpaca JSONL with Mathpix, OCR, GPT tagging

import os
import re
import io
import fitz  # PyMuPDF
import json
import time
import base64
import requests
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from datetime import timedelta, datetime
from collections import defaultdict

# Load API keys from .env in scripts directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Adjusted Paths
PDF_SLIDES_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../materials/Slides"))
PDF_TEXTBOOK_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../materials/TextBook"))
OUTPUT_IMAGES = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output_images"))
OUTPUT_JSONL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_train.jsonl"))
os.makedirs(OUTPUT_IMAGES, exist_ok=True)

# -------------- MATHPIX Convert API -------------- #
def upload_pdf_convert_api(file_path):
    url = "https://api.mathpix.com/v3/pdf"
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "options_json": json.dumps({
                "formats": ["text", "latex_styled", "text+latex", "json"],
                "output_format": "json",
                "math_inline_delims": ["$", "$"],
                "math_display_delims": ["$$", "$$"]
            })
        }
        headers = {
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY
        }
        res = requests.post(url, headers=headers, files=files, data=data)
        return res.json().get("pdf_id")

def poll_pdf_result(pdf_id):
    url = f"https://api.mathpix.com/v3/pdf/{pdf_id}"
    headers = {"app_id": MATHPIX_APP_ID, "app_key": MATHPIX_APP_KEY}
    with tqdm(total=100, desc="Processing PDF with Mathpix", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        while True:
            res = requests.get(url, headers=headers)
            result = res.json()
            if result.get("status") == "completed":
                pbar.n = 100
                pbar.refresh()

                # ✅ New logic to extract text from structured JSON
                try:
                    pages = result.get("json", {}).get("pages", [])
                    full_text = "\n\n".join(page.get("text", "") for page in pages)
                    if not full_text.strip():
                        print("[!] Mathpix finished but returned empty structured JSON content.")
                    return full_text
                except Exception as e:
                    print(f"[!] Failed to parse JSON text from Mathpix: {e}")
                    return ""
            elif result.get("status") == "error":
                raise Exception(f"[✗] Mathpix error: {result.get('error', 'Unknown error')}")
            time.sleep(3)
            pbar.update(5 if pbar.n < 95 else 0)

# -------------- Local OCR / Image fallback -------------- #
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        pix = doc.load_page(page_num).get_pixmap(dpi=300)
        img_path = os.path.join(OUTPUT_IMAGES, f"page_{page_num}.png")
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

# -------------- GPT-4V Diagram Description -------------- #
def gpt4v_image_prompt(image_path, context="Explain this diagram in EE 140 context"):
    if not OPENAI_API_KEY:
        print("[!] GPT-4V disabled: OPENAI_API_KEY not found.")
        return ""

    try:
        with open(image_path, "rb") as img:
            b64_img = base64.b64encode(img.read()).decode()

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {"role": "system", "content": "You're an expert tutor who explains diagrams clearly."},
                {"role": "user", "content": [
                    {"type": "text", "text": context},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ],
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()

        result = res.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")

    except Exception as e:
        print(f"[!] GPT-4V diagram interpretation failed: {str(e)}")
        return ""


# -------------- Tagging / Cleaning Helpers -------------- #
def classify_tags(text):
    tags = []
    low = text.lower()
    if any(k in low for k in ["divergence", "curl", "laplacian", "gradient"]): tags.append("electromagnetics")
    if any(k in low for k in ["cylindrical", "spherical"]): tags.append("coordinate")
    if any(k in low for k in ["∂", "integral", "derivative"]): tags.append("calculus")
    if any(k in low for k in ["diagram explanation"]): tags.append("visual_reasoning")
    return tags if tags else ["other"]

def clean_and_group_blocks(text):
    blocks = text.split("\n\n")
    grouped, current = [], []
    for line in blocks:
        line = line.strip()
        if re.search(r"[=+\-*/^\\]", line): current.append(line)
        else:
            if current: grouped.append(" \n".join(current)); current = []
            grouped.append(line)
    if current: grouped.append(" \n".join(current))
    return grouped

# -------------- Final JSONL Writer -------------- #
def write_jsonl(entries, output_path):
    with open(output_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

# -------------- Master Runner -------------- #
def process_pdf(pdf_path, tag_counter):
    print(f"[+] Processing {pdf_path}")
    entries = []
    try:
        pdf_id = upload_pdf_convert_api(pdf_path)
        text = poll_pdf_result(pdf_id)
    except Exception as e:
        print("[!] Convert API failed, falling back to local OCR...")
        text = ""
        for i, image_path in convert_pdf_to_images(pdf_path):
            text += mathpix_image_ocr(image_path)
            text += "\n\n"
            diagram_desc = gpt4v_image_prompt(image_path)
            if diagram_desc:
                text += f"[Diagram Explanation]\n{diagram_desc}\n\n"
    blocks = clean_and_group_blocks(text)
    for i, block in enumerate(blocks):
        if len(block.strip()) < 10:
            continue
        tags = classify_tags(block)
        for tag in tags:
            tag_counter[tag] += 1
        entry = {
            "instruction": "Explain or derive the following expression or concept from EE 140 class:",
            "input": "",
            "output": block.strip(),
            "meta": {
                "source": os.path.basename(pdf_path),
                "line": i,
                "tags": tags
            }
        }
        entries.append(entry)
    return entries

# -------------- Entry Point -------------- #
if __name__ == "__main__":
    all_entries = []
    tag_counter = defaultdict(int)
    start_time = datetime.now()

    folders = [PDF_SLIDES_FOLDER, PDF_TEXTBOOK_FOLDER]
    pdf_files = []
    for folder in folders:
        if os.path.exists(folder):
            pdf_files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")])

    with tqdm(total=len(pdf_files), desc="Total Progress", unit="pdf") as overall:
        for pdf_path in pdf_files:
            all_entries.extend(process_pdf(pdf_path, tag_counter))
            overall.update(1)

    write_jsonl(all_entries, OUTPUT_JSONL)
    end_time = datetime.now()
    duration = str(timedelta(seconds=int((end_time - start_time).total_seconds())))
    print(f"[✓] Done! {len(all_entries)} entries saved to {OUTPUT_JSONL} in {duration}")

    print("\n📊 Summary by Tag:")
    for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
        stars = "★" * min(5, count // 10) + "☆" * max(0, 5 - count // 10)
        print(f"  - {tag:20s}: {count:4d} entries  | Estimated Accuracy: {stars}")

