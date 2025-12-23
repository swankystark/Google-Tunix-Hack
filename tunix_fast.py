# Fast Tunix preprocessing - optimized for speed
from datasets import load_dataset
import json, os, re
from typing import List, Dict, Any, Optional

os.makedirs("tunix_data_preprocessed", exist_ok=True)

def first_present(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def is_trivial_explanation(text: Optional[str]) -> bool:
    if not text:
        return True
    s = str(text).strip()
    if len(s) < 20:
        return True
    return False

def format_example_to_tunix(example, input_keys, reasoning_keys, answer_keys):
    """Fast formatting without complex choice logic."""
    input_text = first_present(example, input_keys) or ""
    reasoning = first_present(example, reasoning_keys) or ""
    answer = first_present(example, answer_keys) or ""

    if isinstance(input_text, (list, dict)):
        input_text = json.dumps(input_text, ensure_ascii=False)
    if isinstance(reasoning, (list, dict)):
        reasoning = json.dumps(reasoning, ensure_ascii=False)
    if isinstance(answer, (list, dict)):
        answer = json.dumps(answer, ensure_ascii=False)

    input_text = str(input_text).strip()[:2000]  # Limit input
    reasoning = str(reasoning).strip()[:4000]  # Limit reasoning
    answer = str(answer).strip()[:1000]  # Limit answer

    user_prefix = "<start_of_turn>user\nSolve the following problem:\n\n"
    user_suffix = "\n<end_of_turn>"
    model_prefix = "<start_of_turn>model\n"
    model_suffix = "\n<end_of_turn>"

    full_text = (
        f"{user_prefix}{input_text}{user_suffix}\n"
        f"{model_prefix}<reasoning>\n{reasoning}\n</reasoning>\n"
        f"<answer>\n{answer}\n</answer>{model_suffix}"
    )
    return {"text": full_text}

def fast_sample_export(dataset_id, split, sample_count, outpath, input_keys, reasoning_keys, answer_keys):
    """Simplified, faster export function."""
    try:
        print(f"Loading {dataset_id}...", end=" ", flush=True)
        ds = load_dataset(dataset_id, split=split)
        print(f"({len(ds)} total)")

        n_select = min(len(ds), sample_count * 2)  # Less oversampling
        selected = ds.shuffle(seed=42).select(range(n_select))

        written = 0
        with open(outpath, "w", encoding="utf-8") as fout:
            for ex in selected:
                try:
                    reasoning = first_present(ex, reasoning_keys) or ""
                    if not reasoning or len(str(reasoning).strip()) < 30:
                        continue
                    if is_trivial_explanation(reasoning):
                        continue

                    mapped = format_example_to_tunix(ex, input_keys, reasoning_keys, answer_keys)
                    fout.write(json.dumps({"text": mapped["text"]}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= sample_count:
                        break
                except:
                    continue

        print(f"  ✓ Wrote {written} examples to {os.path.basename(outpath)}")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

# Run datasets
print("\n=== TUNIX Dataset Preprocessing ===\n")

total = 0

# Math: 40k
total += fast_sample_export(
    "ndavidson/sat-math-chain-of-thought", "train", 40000,
    "tunix_data_preprocessed/sat_math_cot.jsonl",
    ["question", "prompt"], ["reasoning", "explanation"], ["answer", "label"]
)

# Coding: 25k
total += fast_sample_export(
    "open-r1/codeforces-cots", "train", 15000,
    "tunix_data_preprocessed/codeforces_cots.jsonl",
    ["problem", "statement"], ["chain_of_thought", "reasoning"], ["solution", "code"]
)

total += fast_sample_export(
    "HuggingFaceH4/Code-Feedback", "train", 10000,
    "tunix_data_preprocessed/code_feedback.jsonl",
    ["code_snippet", "submission"], ["feedback", "explanation"], ["score", "label"]
)

# Science: 16k
total += fast_sample_export(
    "Locutusque/arc-cot", "train", 8000,
    "tunix_data_preprocessed/arc_cot.jsonl",
    ["question", "stem"], ["explanation", "reasoning"], ["answer", "label"]
)

total += fast_sample_export(
    "prithivmlmods/Turing-Reason-CoT", "train", 8000,
    "tunix_data_preprocessed/turing_reason_cot.jsonl",
    ["question", "input"], ["chain_of_thought", "explanation"], ["solution", "answer"]
)

# Creative ideation: 7k
total += fast_sample_export(
    "isaiahbjork/chain-of-thought", "train", 7000,
    "tunix_data_preprocessed/isaiah_chain_of_thought.jsonl",
    ["prompt", "input"], ["reasoning", "explanation"], ["completion", "answer"]
)

# Other: 20k
total += fast_sample_export(
    "prithivMLmods/Demeter-LongCoT-6M", "train", 10000,
    "tunix_data_preprocessed/demeter_longcot.jsonl",
    ["instruction", "prompt"], ["long_reasoning", "reasoning"], ["response", "answer"]
)

total += fast_sample_export(
    "prithivMLmods/Atlas-Think-CoT-12M", "train", 10000,
    "tunix_data_preprocessed/atlas_think_cot.jsonl",
    ["prompt", "instruction"], ["reasoning", "chain_of_thought"], ["response", "answer"]
)

print(f"\n=== Complete ===")
print(f"Total examples processed: {total:,}")
print(f"Output directory: tunix_data_preprocessed/")
print(f"Files created: {len(os.listdir('tunix_data_preprocessed'))}")
