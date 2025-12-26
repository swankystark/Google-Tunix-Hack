import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_ID = "google/gemma-3-1b-it"
INPUT_FILE = "output/qwill_creative_writing_raw.jsonl"
OUTPUT_DIR = "output/processed_creative_writing_dataset"
SYSTEM_PROMPT = "Solve step-by-step. Use <reasoning> for calculations and logic, and <answer> for the final result."
MAX_TOKENS = 2048
LIMIT_ROWS = None  # Full run

# Load tokenizer once at the start (no fallback, mirrors STF)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def _extract_think_and_answer(text: str):
    """Extracts <think>...</think> as reasoning and <answer>...</answer> as final."""
    reasoning = ""
    answer = ""

    start_t = text.find("<think>")
    end_t = text.find("</think>")
    if start_t != -1 and end_t != -1 and end_t > start_t:
        reasoning = text[start_t + len("<think>"): end_t].strip()

    start_a = text.find("<answer>")
    end_a = text.find("</answer>")
    if start_a != -1 and end_a != -1 and end_a > start_a:
        answer = text[start_a + len("<answer>"): end_a].strip()
    elif end_t != -1:
        answer = text[end_t + len("</think>"):].strip()
    else:
        answer = text.strip()

    return reasoning, answer

def process_and_filter(row):
    """Converts to messages and checks token length (STF-style)."""
    prompt = row.get("prompt", "")
    raw_text = row.get("gemini", "")

    reasoning, final_answer = _extract_think_and_answer(raw_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{final_answer}\n</answer>"}
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_count = len(tokenizer.encode(full_text))

    if token_count <= MAX_TOKENS:
        return {"messages": messages, "length": token_count}
    return None

def main():
    print(f"Starting preprocessing and filtering (Max: {MAX_TOKENS} tokens)...")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    processed_rows = []
    skipped_count = 0
    total_count = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if LIMIT_ROWS and i >= LIMIT_ROWS:
                break
            
            total_count += 1
            row = json.loads(line)
            result = process_and_filter(row)
            
            if result:
                processed_rows.append(result)
            else:
                skipped_count += 1
            
            if i % 100 == 0 and i > 0:
                print(f"Processed: {i} | Kept: {len(processed_rows)} | Skipped: {skipped_count}")
    
    # Create dataset and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = Dataset.from_list(processed_rows)
    dataset.save_to_disk(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(f"--- Done! ---")
    print(f"Total Rows Processed: {total_count}")
    print(f"Final Dataset Size: {len(dataset)} rows")
    print(f"Total Rows Removed: {skipped_count} ({skipped_count/total_count*100:.1f}%)")
    print(f"Saved to: {OUTPUT_DIR}")
    
    # Token statistics
    if len(dataset) > 0:
        lengths = [row['length'] for row in dataset]
        print(f"\nToken Statistics:")
        print(f"  Max: {max(lengths)}")
        print(f"  Min: {min(lengths)}")
        print(f"  Avg: {sum(lengths)/len(lengths):.0f}")

if __name__ == "__main__":
    main()
