import json
import os
from datasets import Dataset
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_ID = "google/gemma-3-1b-it"
INPUT_FILE = "raw_by_source/olympiads.jsonl"
OUTPUT_DIR = "processed_math_dataset"
SYSTEM_PROMPT = "Solve step-by-step. Use <reasoning> for calculations and logic, and <answer> for the final result."
MAX_TOKENS = 2048 
LIMIT_ROWS = 5 # Set to None for full 125K run

# Load tokenizer once at the start
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def extract_boxed_answer(solution):
    last_boxed_idx = solution.rfind("\\boxed{")
    if last_boxed_idx == -1: return "No boxed answer found"
    start_idx = last_boxed_idx + len("\\boxed{")
    depth = 1
    for i in range(start_idx, len(solution)):
        if solution[i] == '{': depth += 1
        elif solution[i] == '}':
            depth -= 1
            if depth == 0: return solution[start_idx:i]
    return "No boxed answer found"

def process_and_filter(row):
    """Converts to messages AND checks token length."""
    solution = row.get("solution", "")
    answer = extract_boxed_answer(solution)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row.get("problem", "")},
        {"role": "assistant", "content": f"<reasoning>\n{solution}\n</reasoning>\n<answer>\n{answer}\n</answer>"}
    ]
    
    # 1. Apply the chat template to see the REAL final string
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # 2. Count tokens
    token_count = len(tokenizer.encode(full_text))
    
    if token_count <= MAX_TOKENS:
        return {"messages": messages, "length": token_count}
    else:
        return None # Signal for removal

def main():
    print(f"Starting preprocessing and filtering (Max: {MAX_TOKENS} tokens)...")
    processed_rows = []
    skipped_count = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if LIMIT_ROWS and i >= LIMIT_ROWS: break
            
            row = json.loads(line)
            result = process_and_filter(row)
            
            if result:
                processed_rows.append(result) 
            else:
                skipped_count += 1
            
            if i % 1000 == 0:
                print(f"Processed: {i} | Skipped: {skipped_count}")

    dataset = Dataset.from_list(processed_rows)
    dataset.save_to_disk(OUTPUT_DIR)
    
    print(f"\n--- Done! ---")
    print(f"Final Dataset Size: {len(dataset)} rows")
    print(f"Total Rows Removed: {skipped_count}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()