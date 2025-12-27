from datasets import load_dataset
import json
import os

# ================= CONFIG =================
DATASET_NAME = "open-r1/codeforces-cots"
SPLIT = "train"
OUTPUT_DIR = "streamed_coding_cot"
ROWS_TO_DOWNLOAD = 30_000
CHECKPOINT_FILE = "checkpoint.txt"
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load last checkpoint (if exists) ----
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        start_row = int(f.read().strip())
else:
    start_row = 0

end_row = start_row + ROWS_TO_DOWNLOAD
output_path = f"{OUTPUT_DIR}/coding_cot_{start_row}_{end_row}.jsonl"

print(f"Starting from row: {start_row}")
print(f"Saving until row: {end_row}")

# ---- Stream dataset ----
dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
    streaming=True
)

count = 0
current_row = 0

with open(output_path, "w", encoding="utf-8") as out_file:
    for example in dataset:
        if current_row < start_row:
            current_row += 1
            continue

        if count >= ROWS_TO_DOWNLOAD:
            break

        out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
        count += 1
        current_row += 1

# ---- Save checkpoint ----
with open(CHECKPOINT_FILE, "w") as f:
    f.write(str(current_row))

print(f"Downloaded {count} rows")
print(f"Checkpoint saved at row {current_row}")
print(f"Data saved to {output_path}")
