import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

# -------- CONFIG --------
MODEL_ID = "google/gemma-3-1b-it"
INPUT_FILE = "streamed_coding_cot/coding_cot_0_30000.jsonl"
OUTPUT_DIR = "processed_coding_dataset"
MAX_TOKENS = 2048
TARGET_ROWS = 10_000

SYSTEM_PROMPT = (
    "You are a competitive programming assistant. "
    "Solve the problem step by step and explain your reasoning clearly."
)
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def process_and_filter(row):
    """
    Convert Codeforces-CoT row to Gemma chat format
    and enforce token limit.
    """
    prompt = row.get("prompt", "")
    generation = row.get("generation", "")

    # Drop empty rows
    if not prompt or not generation:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": f"<reasoning>\n{generation}\n</reasoning>"
        }
    ]

    # Apply Gemma chat template
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    token_count = len(tokenizer.encode(full_text))

    if token_count <= MAX_TOKENS:
        return {
            "messages": messages,
            "length": token_count
        }
    else:
        return None


def main():
    print("Starting preprocessing...")
    print(f"Target rows: {TARGET_ROWS}")
    print(f"Max tokens: {MAX_TOKENS}")

    processed = []
    skipped = 0
    seen = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if len(processed) >= TARGET_ROWS:
                break

            row = json.loads(line)
            seen += 1

            result = process_and_filter(row)
            if result:
                processed.append(result)
            else:
                skipped += 1

            if seen % 1000 == 0:
                print(
                    f"Read: {seen} | "
                    f"Kept: {len(processed)} | "
                    f"Skipped: {skipped}"
                )

    dataset = Dataset.from_list(processed)
    dataset.save_to_disk(OUTPUT_DIR)

    print("\n--- DONE ---")
    print(f"Rows read: {seen}")
    print(f"Final dataset size: {len(dataset)}")
    print(f"Rows skipped: {skipped}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
