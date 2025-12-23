# Tunix / Gemma dataset loading, sampling, cleaning and export notebook
# Produces cleaned JSONL files formatted for Gemma: each example -> single {"text": "<start_of_turn>user ..."}
#
# Requirements:
# pip install datasets huggingface_hub transformers
# Run on Kaggle/Colab with internet. Adjust SAMPLE_* counts to fit TPU/compute budget.

from datasets import load_dataset
import json, os, re
from typing import List, Dict, Any, Optional
from huggingface_hub import login

# Authenticate with HuggingFace
# Token removed for security - set HF_TOKEN environment variable instead
# login(token="YOUR_TOKEN_HERE", add_to_git_credential=False)

os.makedirs("tunix_data_preprocessed", exist_ok=True)

# -------------------- Utilities --------------------

def first_present(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def normalize_choice_answer(label, choices):
    """Return (letter, text) for a label and list of choices.
    label may be index (int), letter ('A','B'...), or text that matches a choice.
    """
    if label is None:
        return (None, "")
    try:
        if isinstance(label, int):
            idx = label
            if 0 <= idx < len(choices):
                return (chr(65 + idx), str(choices[idx]))
    except Exception:
        pass
    lab = str(label).strip()
    if len(lab) == 1 and 'A' <= lab.upper() <= 'Z':
        idx = ord(lab.upper()) - 65
        if 0 <= idx < len(choices):
            return (lab.upper(), str(choices[idx]))
    try:
        idx = int(lab)
        if 0 <= idx < len(choices):
            return (chr(65 + idx), str(choices[idx]))
    except Exception:
        pass
    for i, c in enumerate(choices):
        try:
            if str(lab).strip().lower() == str(c).strip().lower():
                return (chr(65 + i), str(c))
        except Exception:
            continue
    return (None, str(label))

def is_trivial_explanation(text: Optional[str]) -> bool:
    """Detect trivial 'explanations' like 'The answer is (A)' or very short strings."""
    if not text:
        return True
    s = str(text).strip()
    if len(s) < 20:
        # treat extremely short explanations as trivial (configurable)
        return True
    # patterns such as: "The answer is (A)." or "Answer: A" or just "A"
    if re.match(r"^\s*(the\s+answer\s+is\b|answer[:\-]?\s*)[\(\[]?[A-Za-z0-9]+[\)\]]?\s*\.?\s*$", s, flags=re.IGNORECASE):
        return True
    if re.match(r"^[A-Za-z0-9]+\.?$", s):
        return True
    return False

# -------------------- Formatter --------------------

def format_example_to_tunix(example: Dict[str, Any],
                             input_keys: List[str],
                             reasoning_keys: List[str],
                             answer_keys: List[str],
                             choices_key: Optional[str] = None,
                             answer_label_key: Optional[str] = None) -> Dict[str, str]:
    """
    Returns: {"text": full_conversation_string, "reasoning": reasoning_str, "answer": answer_str}
    Full conversation uses Gemma-friendly wrapper:
    <start_of_turn>user
    Solve the following problem... {input_block}
    <end_of_turn>
    <start_of_turn>model
    <reasoning>...</reasoning>
    <answer>...</answer>
    <end_of_turn>
    """
    # input/question
    input_text = first_present(example, input_keys) or ""
    # choices
    choices = None
    if choices_key and choices_key in example and example[choices_key] is not None:
        choices = example[choices_key]
    if choices is not None and not isinstance(choices, list):
        try:
            choices = list(choices)
        except Exception:
            choices = None

    # compose input_block (include choices labeled)
    input_parts = []
    if isinstance(input_text, str) and input_text.strip() != "":
        input_parts.append(str(input_text).strip())
    elif isinstance(input_text, (list, dict)):
        input_parts.append(json.dumps(input_text, ensure_ascii=False))
    if choices:
        opt_lines = []
        for i, c in enumerate(choices):
            try:
                opt_text = str(c).strip()
            except Exception:
                opt_text = json.dumps(c, ensure_ascii=False)
            opt_lines.append(f"{chr(65 + i)}. {opt_text}")
        input_parts.append("Options:\n" + "\n".join(opt_lines))
    input_block = "\n\n".join(input_parts).strip()

    # reasoning & answer extraction
    reasoning = first_present(example, reasoning_keys) or ""
    answer = first_present(example, answer_keys) or ""

    # resolve MCQ labels -> letter + text
    if choices and answer_label_key and (answer is None or str(answer).strip() == ""):
        label = example.get(answer_label_key)
        letter, text_choice = normalize_choice_answer(label, choices)
        if letter:
            answer = f"{letter} | {text_choice}"
        else:
            answer = str(label)
    elif choices and (str(answer).strip() != ""):
        letter, text_choice = normalize_choice_answer(answer, choices)
        if letter:
            answer = f"{letter} | {text_choice}"
        else:
            answer = str(answer)

    reasoning = reasoning if isinstance(reasoning, str) else json.dumps(reasoning, ensure_ascii=False)
    answer = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)

    # Gemma-3 instruction-aware wrapper (user + model turns)
    user_prefix = ("<start_of_turn>user\n"
                   "Solve the following problem. Provide your step-by-step logic inside <reasoning> tags, "
                   "and the final concise result inside <answer> tags.\n\n")
    user_suffix = "\n<end_of_turn>"
    model_prefix = "<start_of_turn>model\n"
    model_suffix = "\n<end_of_turn>"

    full_text = (
        f"{user_prefix}{input_block}{user_suffix}\n"
        f"{model_prefix}<reasoning>\n{reasoning.strip()}\n</reasoning>\n"
        f"<answer>\n{answer.strip()}\n</answer>{model_suffix}"
    )

    return {"text": full_text, "reasoning": reasoning.strip(), "answer": answer.strip()}

# -------------------- Sampling + Cleaning + Export --------------------

def sample_and_export(dataset_id: str,
                      split: str,
                      sample_count: int,
                      outpath: str,
                      input_keys: List[str],
                      reasoning_keys: List[str],
                      answer_keys: List[str],
                      choices_key: Optional[str] = None,
                      answer_label_key: Optional[str] = None,
                      shuffle_seed: int = 42,
                      min_reasoning_chars: int = 50,
                      clean_trivial_explanations: bool = True,
                      max_retries: int = 3):
    """
    Load dataset, oversample to allow filtering, and write cleaned JSONL of sample_count examples.
    Filtering:
    - reasoning length >= min_reasoning_chars
    - remove trivial explanations like 'The answer is (A)' if clean_trivial_explanations=True
    """
    for attempt in range(max_retries):
        try:
            print(f"Loading {dataset_id} {split} (attempt {attempt + 1}/{max_retries}) ...")
            ds = load_dataset(dataset_id, split=split)
            print("Dataset size:", len(ds))
            break
        except Exception as e:
            print(f"Error loading dataset (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to load {dataset_id} after {max_retries} attempts. Skipping...")
                return
            continue

    # oversample (x3) to account for discarded examples during cleaning
    n_select = min(len(ds), sample_count * 3)
    selected = ds.shuffle(seed=shuffle_seed).select(range(n_select))

    written = 0
    with open(outpath, "w", encoding="utf-8") as fout:
        for ex in selected:
            try:
                mapped = format_example_to_tunix(ex, input_keys, reasoning_keys, answer_keys, choices_key, answer_label_key)
                reasoning = mapped.get("reasoning", "")
                if reasoning is None:
                    continue
                if len(reasoning.strip()) < min_reasoning_chars:
                    continue
                if clean_trivial_explanations and is_trivial_explanation(reasoning):
                    continue
                fout.write(json.dumps({"text": mapped["text"]}, ensure_ascii=False) + "\n")
                written += 1
                if written >= sample_count:
                    break
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
    print(f"Wrote {written} cleaned rows -> {outpath}")

# -------------------- Dataset sample configs (adjust samples to suit compute) --------------------

sample_and_export(
    dataset_id="ndavidson/sat-math-chain-of-thought",
    split="train",
    sample_count=40000,
    outpath="tunix_data_preprocessed/sat_math_cot.jsonl",
    input_keys=["question", "prompt", "problem"],
    reasoning_keys=["reasoning", "rationale", "explanation"],
    answer_keys=["answer", "final_answer", "label"],
    choices_key="choices",
    answer_label_key="label",
    min_reasoning_chars=30,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="kaist-ai/CoT-Collection",
    split="train",
    sample_count=20000,
    outpath="tunix_data_preprocessed/kaist_cot_collection.jsonl",
    input_keys=["source", "input", "prompt"],
    reasoning_keys=["rationale", "reasoning", "explanation"],
    answer_keys=["target", "output", "answer"],
    min_reasoning_chars=50,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="prithivMLmods/Atlas-Think-CoT-12M",
    split="train",
    sample_count=10000,
    outpath="tunix_data_preprocessed/atlas_think_cot.jsonl",
    input_keys=["prompt", "instruction", "input"],
    reasoning_keys=["reasoning", "rationale", "chain_of_thought", "explanation"],
    answer_keys=["response", "target", "answer"],
    min_reasoning_chars=80,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="prithivmlmods/Demeter-LongCoT-6M",
    split="train",
    sample_count=10000,
    outpath="tunix_data_preprocessed/demeter_longcot.jsonl",
    input_keys=["instruction", "prompt", "input"],
    reasoning_keys=["long_reasoning", "reasoning", "explanation", "rationale"],
    answer_keys=["response", "answer", "target"],
    min_reasoning_chars=200,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="prithivmlmods/Turing-Reason-CoT",
    split="train",
    sample_count=8000,
    outpath="tunix_data_preprocessed/turing_reason_cot.jsonl",
    input_keys=["question", "prompt", "input"],
    reasoning_keys=["chain_of_thought", "rationale", "explanation"],
    answer_keys=["solution", "answer", "target"],
    min_reasoning_chars=80,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="Locutusque/arc-cot",
    split="train",
    sample_count=8000,
    outpath="tunix_data_preprocessed/arc_cot.jsonl",
    input_keys=["question", "prompt", "stem"],
    reasoning_keys=["explanation", "reasoning", "rationale"],
    answer_keys=["answer", "label", "final_answer"],
    choices_key="choices",
    answer_label_key="answer",
    min_reasoning_chars=40,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="isaiahbjork/chain-of-thought",
    split="train",
    sample_count=7000,
    outpath="tunix_data_preprocessed/isaiah_chain_of_thought.jsonl",
    input_keys=["prompt", "input", "question"],
    reasoning_keys=["reasoning", "rationale", "explanation"],
    answer_keys=["completion", "answer", "target"],
    min_reasoning_chars=50,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="open-r1/codeforces-cots",
    split="train",
    sample_count=15000,
    outpath="tunix_data_preprocessed/codeforces_cots.jsonl",
    input_keys=["problem", "statement", "prompt"],
    reasoning_keys=["chain_of_thought", "reasoning", "rationale"],
    answer_keys=["solution", "final_answer", "code"],
    min_reasoning_chars=80,
    clean_trivial_explanations=True
)

sample_and_export(
    dataset_id="HuggingFaceH4/Code-Feedback",
    split="train",
    sample_count=10000,
    outpath="tunix_data_preprocessed/code_feedback.jsonl",
    input_keys=["code_snippet", "code", "submission"],
    reasoning_keys=["feedback", "explanation", "reasoning"],
    answer_keys=["final_feedback", "score", "label", "suggestion"],
    min_reasoning_chars=40,
    clean_trivial_explanations=True
)

# -------------------- Merge categories (optional) --------------------
CATEGORY_MAP = {
    "creative_writing": ["tunix_data_preprocessed/kaist_cot_collection.jsonl"],
    "creative_ideation": ["tunix_data_preprocessed/kaist_cot_collection.jsonl", "tunix_data_preprocessed/isaiah_chain_of_thought.jsonl"],
    "summarization": ["tunix_data_preprocessed/kaist_cot_collection.jsonl"],
    "math": ["tunix_data_preprocessed/sat_math_cot.jsonl"],
    "coding": ["tunix_data_preprocessed/codeforces_cots.jsonl", "tunix_data_preprocessed/code_feedback.jsonl"],
    "basic_science": ["tunix_data_preprocessed/arc_cot.jsonl", "tunix_data_preprocessed/turing_reason_cot.jsonl"],
    "other": ["tunix_data_preprocessed/atlas_think_cot.jsonl", "tunix_data_preprocessed/demeter_longcot.jsonl", "tunix_data_preprocessed/kaist_cot_collection.jsonl"]
}

def merge_category_files(category: str, file_list: List[str], outpath: str, max_examples: int = 30000):
    seen = 0
    with open(outpath, "w", encoding="utf-8") as fout:
        for f in file_list:
            if not os.path.exists(f):
                print("Warning: missing", f)
                continue
            with open(f, "r", encoding="utf-8") as fin:
                for line in fin:
                    if seen >= max_examples:
                        break
                    fout.write(line)
                    seen += 1
            if seen >= max_examples:
                break
    print(f"Merged {seen} examples for category {category} -> {outpath}")

for cat, files in CATEGORY_MAP.items():
    merge_category_files(cat, files, f"tunix_data_preprocessed/category_{cat}.jsonl", max_examples=30000)

# -------------------- Optional tokenization helper --------------------
# Default max_length = 4096 (recommended). If you have less memory, reduce to 2048 but avoid 512.
try:
    from transformers import AutoTokenizer

    def tokenize_jsonl(infile: str, outfile: str, tokenizer_name: str = "google/flan-t5-base", max_length: int = 4096):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
            for line in fin:
                row = json.loads(line)
                text = row["text"]
                enc = tokenizer(text, truncation=True, max_length=max_length)
                json.dump(enc, fout)
                fout.write("\n")
    print("Tokenization helper ready. Default max_length=4096.")

except Exception as e:
    print("Tokenization helper unavailable (transformers missing or other error):", e)

print("Done. Cleaned files in tunix_data_preprocessed/")
