#!/usr/bin/env python3
"""
Raw downloader for AI-MO/NuminaMath-CoT

- Saves each subdivision (source) into its own JSONL file
- NO filtering, NO Gemini, NO validation
- Rows saved exactly as-is
- Memory-safe (streaming)
"""

import os
import json
import logging
from datasets import load_dataset

# ===================== CONFIG =====================

OUTPUT_DIR = "raw_by_source"

SOURCE_LIMITS = {
    "amc_aime": 4072,
    "olympiads": 12000,
    "aops_forum": 6000,
    "synthetic_amc": 4000,
    "gsm8k": 2000,
    "math": 1928
}

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "raw_download.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logging.getLogger().addHandler(logging.StreamHandler())

# ===================== MAIN =====================

def main():
    logging.info("Starting raw dataset download (per subdivision files)")

    ds = load_dataset(
        "AI-MO/NuminaMath-CoT",
        split="train",
        streaming=True,
    )

    collected = {k: 0 for k in SOURCE_LIMITS}

    # Open one file per subdivision
    files = {
        src: open(os.path.join(OUTPUT_DIR, f"{src}.jsonl"), "w", encoding="utf-8")
        for src in SOURCE_LIMITS
    }

    try:
        for row in ds:
            src = row.get("source")

            # Ignore rows without a valid source
            if src not in SOURCE_LIMITS:
                continue

            # Stop collecting from this source if limit reached
            if collected[src] >= SOURCE_LIMITS[src]:
                continue

            # Write row exactly as-is
            files[src].write(json.dumps(row, ensure_ascii=False) + "\n")
            collected[src] += 1

            # Log progress every 500 rows per source
            if collected[src] % 500 == 0:
                logging.info(f"{src}: collected {collected[src]} rows")

            # Stop completely if all limits reached
            if all(collected[k] >= SOURCE_LIMITS[k] for k in SOURCE_LIMITS):
                logging.info("All subdivision limits reached. Stopping.")
                break

    finally:
        for f in files.values():
            f.close()

        logging.info("Finished raw dataset download")
        logging.info(f"Final counts per subdivision: {collected}")
        logging.info(f"Files saved under: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
