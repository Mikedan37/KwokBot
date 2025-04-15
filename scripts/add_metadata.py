import json
import re

input_file = "../data/kwokbot_train.jsonl"
output_file = "../data/kwokbot_train_tagged.jsonl"

# Basic keyword-based tag system
def tag_instruction(text):
    tags = []

    text_lower = text.lower()

    if any(keyword in text_lower for keyword in ["vector", "dot product", "cross product", "torque", "curl", "projection", "scalar triple"]):
        tags.append("vector")

    if any(keyword in text_lower for keyword in ["coordinate", "cartesian", "cylindrical", "spherical", "polar", "unit vector", "transform"]):
        tags.append("coordinate")

    if any(keyword in text_lower for keyword in ["divergence", "gradient", "laplacian", "∇", "nabla", "operator", "partial derivative", "scale factor"]):
        tags.append("differential_operator")

    if any(keyword in text_lower for keyword in ["electric", "magnetic", "em", "maxwell", "e-field", "b-field", "∇·e", "∇×b"]):
        tags.append("electromagnetics")

    if any(keyword in text_lower for keyword in ["center of mass", "moment of inertia", "mass distribution", "dm", "triangle", "disc", "sphere"]):
        tags.append("mechanics")

    if not tags:
        tags.append("other")

    return tags

# Main script
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            print(f"Skipping empty line at {i}")
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping bad JSON at line {i}: {e}")
            continue

        # Add metadata
        instruction_text = data.get("instruction", "")
        data["meta"] = {
            "source": input_file.split("/")[-1],
            "line": i,
            "tags": tag_instruction(instruction_text)
        }

        outfile.write(json.dumps(data) + "\n")

print(f"🔥 Metadata tagging complete. Output saved to {output_file}")
