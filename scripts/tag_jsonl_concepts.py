
# tag_jsonl_concepts.py - Add spicy concept-level tags to KwokBot training data

import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

INPUT_JSONL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_fallback.jsonl"))
OUTPUT_JSONL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/kwokbot_tagged.jsonl"))

# Spicy concept tagging rules
CONCEPT_TAGS = {
    r"gauss.*law|∇•e|flux|∮e": "gauss_law",
    r"faraday|∇×e|∂b/∂t": "faradays_law",
    r"lhcp": "left_hand_circular_polarization",
    r"rhcp": "right_hand_circular_polarization",
    r"reflection coefficient": "reflection_coefficient",
    r"transmission coefficient": "transmission_coefficient",
    r"z₀": "impedance",
    r"∇×h": "ampere_law",
    r"∇•b": "gauss_magnetic",
    r"wave impedance": "impedance",
    r"β|gamma|propagation": "propagation_constant",
    r"lossy|conductivity|σ": "lossy_medium",
    r"plane wave": "plane_wave",
    r"electric field|e field": "electric_field",
    r"magnetic field|b field": "magnetic_field",
    r"cylindrical": "coordinate_system_cylindrical",
    r"spherical": "coordinate_system_spherical",
    r"cartesian": "coordinate_system_cartesian",
    r"boundary condition": "boundary_conditions",
    r"vector algebra|dot product|cross product": "vector_algebra",
    r"transmission line": "transmission_lines",
    r"matching|impedance match": "impedance_matching",
    r"eigenvalue|eigenvector": "linear_algebra",
    r"divergence|curl|gradient": "vector_operators",
    r"∫|∮|∬": "integration",
    r"∂": "partial_derivative",
    r"∇": "del_operator"
}

def infer_tags(text):
    tags = set()
    lowered = text.lower()
    for pattern, tag in CONCEPT_TAGS.items():
        if re.search(pattern, lowered):
            tags.add(tag)
    return sorted(tags)

def tag_file():
    if not os.path.exists(INPUT_JSONL):
        print("[✗] Input JSONL not found!")
        return

    tagged = []
    with open(INPUT_JSONL, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tagging KwokBot entries with spicy concepts"):
        obj = json.loads(line)
        text = obj.get("output", "")
        tags = infer_tags(text)

        if "meta" not in obj:
            obj["meta"] = {}
        obj["meta"]["concept_tags"] = tags

        tagged.append(obj)

    with open(OUTPUT_JSONL, "w") as out:
        for entry in tagged:
            out.write(json.dumps(entry) + "\n")

    print(f"[🔥] Tagged {len(tagged)} entries and saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    tag_file()
