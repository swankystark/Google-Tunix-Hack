#!/usr/bin/env python3
"""
ROOT CAUSE ANALYSIS & FIX
========================

PROBLEM (0 ROWS PROCESSED):
The original script assumed the KAIST CoT-Collection was a JSON ARRAY:
  [ {example1}, {example2}, ... ]

But the ACTUAL file structure is a JSON DICT with numeric string keys:
  {
    "881087": {source, target, rationale, task, ...},
    "230352": {source, target, rationale, task, ...},
    ...
  }

WHY ORIGINAL SCRIPT FAILED:
1. ijson.items(file, 'item') expects: [ item, item, ... ]
   → No 'item' prefix in dict structure
   → Generator returns 0 objects
   → Loop never executes
   → Script exits with "Processed: 0, Written: 0"

2. File size (2.36GB) is too large to parse as single JSON object
   → But ijson handles this via streaming
   → Only need to change the iterator prefix

SOLUTION: ijson.kvitems(file, '')
- Iterates (key, value) pairs from root-level dict
- '' (empty prefix) = top-level dict
- Each value is one example (source, target, rationale, task)
- Streaming: no full dict loaded into memory
- ~2GB file processed incrementally
"""

import json
import os
import re
import ijson
from pathlib import Path
from huggingface_hub import hf_hub_download

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("./output")
OUTPUT_FILE = OUTPUT_DIR / "kaist_creative_writing.jsonl"
CREATIVE_KEYWORDS = {
    "story", "dialogue", "narrative", "creative", "title", "continuation",
    "paragraph", "fiction", "write", "compose", "generate text", "prompt",
    "ending", "beginning", "scene", "character", "conversation", "abstract"
}
TARGET_COUNT = 15000

# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def is_creative_task(source_text, task_name):
    """Check if task is creative writing."""
    if not source_text:
        return False
    
    combined = (source_text + " " + task_name).lower()
    
    # At least one creative keyword
    if not any(kw in combined for kw in CREATIVE_KEYWORDS):
        return False
    
    # Reject math/logic tasks
    if any(x in combined for x in ["math", "arithmetic", "equation", "solve", "compute"]):
        return False
    
    # Reject QA/trivia
    if any(x in combined for x in ["question", "answer", "trivia", "knowledge"]):
        return False
    
    return True


def looks_like_math(text):
    """Detect mathematical notation."""
    if not text:
        return False
    # Look for math symbols
    return bool(re.search(r'[=+\-*/^]', text[:500]))  # Sample first 500 chars


def is_trivial_rationale(rationale):
    """Filter weak/trivial rationales."""
    if not rationale or len(rationale) < 50:
        return True
    
    lower = rationale.lower().strip()
    
    # Too generic
    if lower.startswith(("the answer is", "because")):
        return True
    
    return False


def format_gemma_conversation(source, target, rationale):
    """
    Format as Gemma-3 turn-based conversation.
    Exact format required for training:
    <start_of_turn>user
    {source}
    <end_of_turn>
    <start_of_turn>model
    <reasoning>
    {rationale}
    </reasoning>
    <answer>
    {target}
    </answer>
    <end_of_turn>
    """
    text = (
        f"<start_of_turn>user\n{source}\n<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"<reasoning>\n{rationale}\n</reasoning>\n"
        f"<answer>\n{target}\n</answer>\n"
        f"<end_of_turn>"
    )
    return text


# ============================================================================
# MAIN STREAMING LOOP (FIXED)
# ============================================================================

def preprocess():
    """
    Main preprocessing pipeline with corrected ijson.kvitems() streaming.
    
    KEY FIX:
    - Original: ijson.items(f, 'item') → fails on dict structure
    - Fixed: ijson.kvitems(f, '') → correctly streams dict key-value pairs
    """
    
    print("=" * 80)
    print("🚀 KAIST CoT-Collection Streaming Preprocessor (FIXED)")
    print("=" * 80)
    
    # Resolve dataset file
    print("\n📥 Resolving dataset file...")
    print("   Downloading from Hugging Face (may take 5-10 minutes first time)...")
    
    try:
        json_file = hf_hub_download(
            repo_id="kaist-ai/CoT-Collection",
            filename="data/CoT_collection_en.json",
            repo_type="dataset"
        )
        print(f"   ✓ File ready: {json_file}")
        file_size_mb = os.path.getsize(json_file) / (1024 ** 2)
        print(f"   Size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        exit(1)
    
    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Stream and filter
    print(f"\n📊 Target: {TARGET_COUNT} creative writing examples")
    print(f"📝 Streaming JSON (FIXED: using ijson.kvitems for dict iteration)...\n")
    
    processed = 0
    written = 0
    errors = 0
    debug_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        try:
            # FIXED: Use ijson.kvitems(f, '') to stream dict key-value pairs
            # This is the CORRECT approach for streaming a single large JSON dict
            with open(json_file, 'rb') as f:
                for key, obj in ijson.kvitems(f, ''):
                    processed += 1
                    
                    # Debug: show first 3 examples
                    if debug_count < 3:
                        print(f"  [DEBUG] Key: {key}")
                        print(f"         Fields: {list(obj.keys()) if isinstance(obj, dict) else 'not dict'}")
                        debug_count += 1
                    
                    # Progress every 500 items
                    if processed % 500 == 0:
                        print(f"  Processed: {processed}, Written: {written}")
                    
                    # Stop if target reached
                    if written >= TARGET_COUNT:
                        print(f"  ✓ Reached target ({TARGET_COUNT} rows)")
                        break
                    
                    # Extract fields
                    if not isinstance(obj, dict):
                        continue
                    
                    source = obj.get("source", "").strip()
                    target = obj.get("target", "").strip()
                    rationale = obj.get("rationale", "").strip()
                    task_name = obj.get("task", "").strip()
                    
                    # Validate presence
                    if not source or not target or not rationale:
                        continue
                    
                    # Filter: creative writing only
                    if not is_creative_task(source, task_name):
                        continue
                    
                    # Filter: no math
                    if looks_like_math(source) or looks_like_math(target):
                        continue
                    
                    # Filter: quality rationale
                    if is_trivial_rationale(rationale):
                        continue
                    
                    # Format and write
                    try:
                        text = format_gemma_conversation(source, target, rationale)
                        row = json.dumps({"text": text}, ensure_ascii=False)
                        out_f.write(row + "\n")
                        written += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            print(f"  ⚠ Write error (row {processed}): {e}")
        
        except ijson.JSONError as e:
            print(f"  ✗ JSON parsing error: {e}")
            print(f"  → File may be corrupted or incomplete")
            exit(1)
        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted by user")
    
    # Final stats
    print(f"\n" + "=" * 80)
    print(f"✅ Processing complete!")
    print(f"📝 Processed: {processed} examples")
    print(f"✨ Written: {written} creative writing examples")
    print(f"💾 Output: {OUTPUT_FILE}")
    print(f"⚠️  Errors: {errors}")
    print(f"=" * 80)


if __name__ == "__main__":
    preprocess()
