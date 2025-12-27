import json
import os
from typing import List

# Config
INPUT_PATH = os.path.join("tunix_data_preprocessed", "creative_ideation.jsonl")
REQUIRED_TAGS: List[str] = [
    "<bos><start_of_turn>user",
    "<end_of_turn>",
    "<start_of_turn>model",
    "<reasoning>",
    "</reasoning>",
    "<answer>",
    "</answer>",
    "<end_of_turn>",
]


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Missing file: {INPUT_PATH}")
        return

    total = 0
    invalid_json = 0
    missing_text_field = 0
    wrong_domain = 0
    missing_tags = 0
    samples = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            text = obj.get("text")
            if not isinstance(text, str):
                missing_text_field += 1
                continue

            if not text.startswith("DOMAIN: creative_ideation"):
                wrong_domain += 1

            if not all(tag in text for tag in REQUIRED_TAGS):
                missing_tags += 1

            if len(samples) < 1:
                samples.append(text)

    print("✅ Verification complete")
    print(f"Total lines: {total}")
    print(f"Invalid JSON lines: {invalid_json}")
    print(f"Missing 'text' field: {missing_text_field}")
    print(f"Wrong domain header: {wrong_domain}")
    print(f"Missing required tags: {missing_tags}")

    if samples:
        s = samples[0]
        print("\n--- First sample (truncated) ---")
        print(s[:800] + ("..." if len(s) > 800 else ""))


if __name__ == "__main__":
    main()
