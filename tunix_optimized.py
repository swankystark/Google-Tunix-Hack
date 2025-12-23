# Optimized Tunix dataset loading with focus on reliable datasets
# Uses only well-maintained datasets without loading scripts

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
    """Return (letter, text) for a label and list of choices."""
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
        return True
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
    """Format example to Gemma-compatible Tunix format."""
    input_text = first_present(example, input_keys) or ""
    choices = None
    if choices_key and choices_key in example and example[choices_key] is not None:
        choices = example[choices_key]
    if choices is not None and not isinstance(choices, list):
        try:
            choices = list(choices)
        except Exception:
            choices = None

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

    reasoning = first_present(example, reasoning_keys) or ""
    answer = first_present(example, answer_keys) or ""

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
                      clean_trivial_explanations: bool = True):
    """Load dataset, sample, and write cleaned JSONL."""
    try:
        print(f"\n=== Loading {dataset_id} ===")
        ds = load_dataset(dataset_id, split=split)
        print(f"Dataset size: {len(ds)}")

        n_select = min(len(ds), sample_count * 3)
        selected = ds.shuffle(seed=shuffle_seed).select(range(n_select))

        written = 0
        with open(outpath, "w", encoding="utf-8") as fout:
            for i, ex in enumerate(selected):
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
                    if i < 10:  # Only print first 10 errors
                        pass
                    continue
        print(f"✓ Wrote {written}/{sample_count} rows -> {outpath}")
        return written
    except Exception as e:
        print(f"✗ Failed to load {dataset_id}: {str(e)[:100]}")
        return 0

# -------------------- Dataset configs with recommended distribution --------------------

# Math: 40k total (SAT-Math focused)
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

# Coding: 25k total (Codeforces 15k + Code-Feedback 10k)
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

# Basic science: 16k total (ARC 8k + Turing 8k)
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

# Creative ideation: 7k
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

# Other domains: 20k (mix of datasets)
sample_and_export(
    dataset_id="prithivMLmods/Demeter-LongCoT-6M",
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

print("\n" + "="*60)
print("✓ Dataset preprocessing complete!")
print("="*60)
