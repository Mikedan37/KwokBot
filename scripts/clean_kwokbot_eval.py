import json
import re

def clean_text(text):
    if not text:
        return ""
    # Remove weird control characters
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(infile):
            try:
                data = json.loads(line)
                data["instruction"] = clean_text(data.get("instruction", ""))
                data["input"] = clean_text(data.get("input", ""))
                data["output"] = clean_text(data.get("output", ""))
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write("\n")
            except json.JSONDecodeError as e:
                print(f"❌ Skipping line {i+1}: JSON error - {e}")

if __name__ == "__main__":
    clean_jsonl("data/kwokbot_eval.jsonl", "data/kwokbot_eval_cleaned.jsonl")
