import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= CONFIG =================
DATASET_ID = "moremilk/General_Inquiry_Thinking-Chain-Of-Thought"
SPLIT = "train"

OUTPUT_DIR = "tunix_data_preprocessed"
OUTPUT_FILE = "creative_ideation.jsonl"

SYSTEM_INSTRUCTION = "Generate creative ideas thoughtfully and coherently."
MAX_ROWS = 8000
MAX_TOKENS = 2048

# =========================================
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

def extract_reasoning(meta_reasoning: str, answer_text: str) -> str:
    reason = meta_reasoning or ""
    # Prefer <think>...</think> inside metadata
    if reason:
        s = reason.find("<think>")
        e = reason.find("</think>")
        if s != -1 and e != -1 and e > s:
            return reason[s + 7 : e].strip()
        return reason.strip()
    # Fallback to <thinking> tags in the answer
    s = answer_text.find("<thinking>")
    e = answer_text.find("</thinking>")
    if s != -1 and e != -1 and e > s:
        return answer_text[s + 10 : e].strip()
    # Last resort: brief generic rationale
    return "I will generate high-quality, useful ideas based on the question."

def format_row(prompt: str, reasoning: str, final_answer: str) -> str:
    return (
        "DOMAIN: creative_ideation\n"
        "<bos><start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{prompt.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        "<reasoning>\n"
        f"{reasoning.strip()}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{final_answer.strip()}\n"
        "</answer><end_of_turn>"
    )

def main():
    print("🚀 Loading Creative Ideation dataset...")
    ds = load_dataset(DATASET_ID, split=SPLIT)

    seen = set()
    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if written >= MAX_ROWS:
                break

            prompt = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            meta = row.get("metadata") or {}
            meta_reasoning = meta.get("reasoning") if isinstance(meta, dict) else ""

            if len(prompt) < 20 or len(answer) < 50:
                skipped += 1
                continue
            if prompt in seen:
                continue
            seen.add(prompt)

            reasoning = extract_reasoning(meta_reasoning or "", answer)
            text = format_row(prompt, reasoning, answer)
            
            # Filter by token count
            token_count = len(tokenizer.encode(text))
            if token_count > MAX_TOKENS:
                skipped += 1
                continue
            
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

            if written % 1000 == 0:
                print(f"Written: {written} | Skipped: {skipped}")

    print("\n✅ DONE")
    print(f"Final rows: {written}")
    print(f"Skipped (token limit): {skipped}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
