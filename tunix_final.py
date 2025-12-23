# Fast Tunix Preprocessing - Final Optimized Version
from datasets import load_dataset
import json, os
from typing import List, Dict, Any, Optional

os.makedirs("tunix_data_preprocessed", exist_ok=True)

def first_present(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def extract_nested_text(obj, max_depth=2, current_depth=0):
    """Recursively extract text from nested structures."""
    if current_depth > max_depth:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ['content', 'text', 'explanation', 'reasoning', 'answer', 'output']:
            if k in obj and obj[k]:
                return extract_nested_text(obj[k], max_depth, current_depth + 1)
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        texts = [extract_nested_text(item, max_depth, current_depth + 1) for item in obj]
        return " ".join([t for t in texts if t])
    return str(obj)

def format_tunix(problem: str, reasoning: str, answer: str) -> str:
    """Format example in Gemma-friendly Tunix format."""
    problem = str(problem).strip()[:2000]
    reasoning = str(reasoning).strip()[:4000]
    answer = str(answer).strip()[:1000]

    return (f"<start_of_turn>user\nSolve the following problem:\n\n{problem}\n<end_of_turn>\n"
            f"<start_of_turn>model\n<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<answer>\n{answer}\n</answer>\n<end_of_turn>")

def process_sat_math():
    """Process SAT Math dataset."""
    try:
        print("Loading SAT Math...", end=" ", flush=True)
        ds = load_dataset('ndavidson/sat-math-chain-of-thought', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/sat_math_cot.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(120000, len(ds)))):
                try:
                    reasoning_chain = ex.get('reasoning_chain', {})
                    if isinstance(reasoning_chain, dict):
                        steps = reasoning_chain.get('steps', [])
                        reasoning = " ".join([s.get('explanation', '') for s in steps if isinstance(s, dict)])
                    else:
                        reasoning = str(reasoning_chain)

                    if len(reasoning.strip()) < 30:
                        continue

                    text = format_tunix(
                        ex.get('question', ''),
                        reasoning,
                        ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 40000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_codeforces():
    """Process Codeforces dataset."""
    try:
        print("Loading Codeforces...", end=" ", flush=True)
        ds = load_dataset('open-r1/codeforces-cots', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/codeforces_cots.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(45000, len(ds)))):
                try:
                    messages = ex.get('messages', [])
                    reasoning = ""
                    if isinstance(messages, list) and len(messages) > 0:
                        # Combine all message content
                        for msg in messages:
                            if isinstance(msg, dict) and 'content' in msg:
                                reasoning += msg['content'] + "\n"

                    if len(reasoning.strip()) < 80:
                        continue

                    text = format_tunix(
                        f"Problem: {ex.get('title', '')}\n{ex.get('description', '')[:500]}",
                        reasoning,
                        ex.get('editorial', '')[:500]
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 15000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_code_feedback():
    """Process Code Feedback dataset."""
    try:
        print("Loading Code Feedback...", end=" ", flush=True)
        ds = load_dataset('HuggingFaceH4/Code-Feedback', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/code_feedback.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(30000, len(ds)))):
                try:
                    code = extract_nested_text(ex.get('code_snippet') or ex.get('submission') or "")
                    feedback = extract_nested_text(ex.get('feedback') or ex.get('explanation') or "")

                    if len(feedback.strip()) < 40:
                        continue

                    text = format_tunix(
                        f"Review code:\n{code[:500]}",
                        feedback,
                        ex.get('label', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 10000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_arc():
    """Process ARC dataset."""
    try:
        print("Loading ARC...", end=" ", flush=True)
        ds = load_dataset('Locutusque/arc-cot', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/arc_cot.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(24000, len(ds)))):
                try:
                    reasoning = extract_nested_text(ex.get('explanation') or ex.get('reasoning') or "")

                    if len(reasoning.strip()) < 40:
                        continue

                    text = format_tunix(
                        ex.get('question', ''),
                        reasoning,
                        ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 8000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_turing():
    """Process Turing Reason dataset."""
    try:
        print("Loading Turing-Reason...", end=" ", flush=True)
        ds = load_dataset('prithivmlmods/Turing-Reason-CoT', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/turing_reason_cot.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(24000, len(ds)))):
                try:
                    reasoning = extract_nested_text(ex.get('chain_of_thought') or ex.get('explanation') or "")

                    if len(reasoning.strip()) < 80:
                        continue

                    text = format_tunix(
                        ex.get('question', ''),
                        reasoning,
                        ex.get('solution', '') or ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 8000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_isaiah():
    """Process Isaiah chain-of-thought dataset."""
    try:
        print("Loading Isaiah CoT...", end=" ", flush=True)
        ds = load_dataset('isaiahbjork/chain-of-thought', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/isaiah_chain_of_thought.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(21000, len(ds)))):
                try:
                    reasoning = extract_nested_text(ex.get('reasoning') or ex.get('explanation') or "")

                    if len(reasoning.strip()) < 50:
                        continue

                    text = format_tunix(
                        ex.get('prompt', ''),
                        reasoning,
                        ex.get('completion', '') or ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 7000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_demeter():
    """Process Demeter LongCoT dataset."""
    try:
        print("Loading Demeter LongCoT...", end=" ", flush=True)
        ds = load_dataset('prithivmlmods/Demeter-LongCoT-6M', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/demeter_longcot.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(30000, len(ds)))):
                try:
                    reasoning = extract_nested_text(ex.get('long_reasoning') or ex.get('reasoning') or "")

                    if len(reasoning.strip()) < 200:
                        continue

                    text = format_tunix(
                        ex.get('instruction', ''),
                        reasoning,
                        ex.get('response', '') or ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 10000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

def process_atlas():
    """Process Atlas-Think-CoT dataset."""
    try:
        print("Loading Atlas-Think-CoT...", end=" ", flush=True)
        ds = load_dataset('prithivMLmods/Atlas-Think-CoT-12M', split='train')
        print(f"({len(ds)} total)")

        written = 0
        with open("tunix_data_preprocessed/atlas_think_cot.jsonl", "w", encoding="utf-8") as f:
            for ex in ds.shuffle(seed=42).select(range(min(30000, len(ds)))):
                try:
                    reasoning = extract_nested_text(ex.get('reasoning') or ex.get('chain_of_thought') or "")

                    if len(reasoning.strip()) < 80:
                        continue

                    text = format_tunix(
                        ex.get('prompt', ''),
                        reasoning,
                        ex.get('response', '') or ex.get('answer', '')
                    )
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= 10000:
                        break
                except:
                    continue

        print(f"  ✓ {written} examples")
        return written
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:60]}")
        return 0

# Main execution
print("\n" + "="*60)
print("TUNIX DATASET PREPROCESSING")
print("="*60 + "\n")

total = 0
total += process_sat_math()
total += process_codeforces()
# Skip Code-Feedback due to download size - focus on other datasets
# total += process_code_feedback()
total += process_arc()
total += process_turing()
total += process_isaiah()
total += process_demeter()
total += process_atlas()

print("\n" + "="*60)
print(f"TOTAL EXAMPLES: {total:,} / 125,000 target")
print("="*60)

# List output files
files = os.listdir("tunix_data_preprocessed")
print(f"\n✓ Created {len(files)} output files:")
for fname in sorted(files):
    fpath = os.path.join("tunix_data_preprocessed", fname)
    lines = len(open(fpath).readlines())
    size_mb = os.path.getsize(fpath) / (1024*1024)
    print(f"  - {fname}: {lines:,} examples ({size_mb:.1f} MB)")
