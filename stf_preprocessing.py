import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# =========================
# CONFIGURATION
# =========================
MODEL_ID = "google/gemma-3-1b-it"
DATASET_ID = "Locutusque/arc-cot"
SPLIT = "train"

OUTPUT_DIR = "processed_arc_cot_gemma_final"

SYSTEM_PROMPT = (
    "Solve step-by-step. Use <reasoning> for calculations and logic, "
    "and <answer> for the final result."
)

MAX_TOKENS = 2048
LIMIT_ROWS = None  # set small number for testing

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# =========================
# HELPERS
# =========================
def extract_final_label(text: str) -> str:
    """
    Extract final MCQ label (A/B/C/D) only.
    """
    patterns = [
        r"final answer\s*[:\-]\s*([A-D])",
        r"answer\s*[:\-]\s*([A-D])",
        r"the answer is\s*([A-D])",
        r"\b([A-D])\b\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return "A"  # safe fallback


def clean_reasoning(text: str) -> str:
    """
    Remove answer leakage like 'A: A. dry palms'
    """
    cleaned = []
    for line in text.splitlines():
        if re.match(r"^[A-D]\s*:\s*[A-D]\.", line.strip()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def process_and_filter(row):
    """
    DO NOT CHANGE TAG STRUCTURE
    """
    question = row["question"]
    raw_reasoning = row["answer"]

    final_label = extract_final_label(raw_reasoning)
    reasoning = clean_reasoning(raw_reasoning)

    reasoning_with_box = f"{reasoning}\n\n\\boxed{{{final_label}}}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": (
                f"<reasoning>\n{reasoning_with_box}\n</reasoning>\n"
                f"<answer>\n{final_label}\n</answer>"
            ),
        },
    ]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    token_count = len(tokenizer.encode(rendered))

    if token_count <= MAX_TOKENS:
        return {"messages": messages, "length": token_count}
    return None


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("Loading Locutusque/arc-cot dataset (parquet-safe)...")

    dataset = load_dataset(DATASET_ID, split=SPLIT)

    processed = []
    skipped = 0

    for i, row in enumerate(dataset):
        if LIMIT_ROWS and i >= LIMIT_ROWS:
            break

        result = process_and_filter(row)
        if result:
            processed.append(result)
        else:
            skipped += 1

        if i % 1000 == 0:
            print(f"Processed: {i} | Skipped: {skipped}")

    final_dataset = Dataset.from_list(processed)
    final_dataset.save_to_disk(OUTPUT_DIR)

    print("\n--- DONE ---")
    print(f"Final dataset size: {len(final_dataset)}")
    print(f"Rows skipped: {skipped}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
