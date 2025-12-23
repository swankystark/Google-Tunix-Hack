# Ultra-Fast Tunix Preprocessing - Lightweight Version
from datasets import load_dataset
import json, os, time
from typing import List, Dict, Any, Optional

os.makedirs("tunix_data_preprocessed", exist_ok=True)

def extract_text(obj):
    """Extract text from various structures."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Try common keys
        for k in ['content', 'text', 'explanation', 'reasoning', 'answer', 'output']:
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return " ".join([extract_text(item) for item in obj if item])
    return str(obj) if obj else ""

def format_tunix(problem: str, reasoning: str, answer: str) -> str:
    """Format in Tunix style."""
    return (f"<start_of_turn>user\n{str(problem).strip()[:1500]}\n<end_of_turn>\n"
            f"<start_of_turn>model\n<reasoning>\n{str(reasoning).strip()[:3000]}\n</reasoning>\n"
            f"<answer>\n{str(answer).strip()[:500]}\n</answer>\n<end_of_turn>")

def quick_export(dataset_id, split, outfile, max_count, extractors):
    """Fast extraction with timeout and simple error handling."""
    try:
        print(f"  {dataset_id}...", end=" ", flush=True)
        start = time.time()
        ds = load_dataset(dataset_id, split=split)
        print(f"({len(ds)})", end=" ", flush=True)

        written = 0
        for i, ex in enumerate(ds.shuffle(seed=42).take(max_count * 2)):
            try:
                problem = extract_text(ex.get(extractors.get('problem', [''])[0], ''))
                reasoning = extract_text(ex.get(extractors.get('reasoning', [''])[0], ''))
                answer = extract_text(ex.get(extractors.get('answer', [''])[0], ''))

                if len(reasoning.strip()) < 30:
                    continue

                with open(outfile, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"text": format_tunix(problem, reasoning, answer)}, ensure_ascii=False) + "\n")
                written += 1
                if written >= max_count:
                    break
            except:
                pass

        elapsed = time.time() - start
        print(f"✓ {written} ({elapsed:.0f}s)")
        return written
    except Exception as e:
        print(f"✗ ({str(e)[:40]})")
        return 0

# Clear output files
for f in os.listdir("tunix_data_preprocessed"):
    if f.endswith(".jsonl"):
        os.remove(os.path.join("tunix_data_preprocessed", f))

print("\n" + "="*60)
print("TUNIX FAST PREPROCESSING")
print("="*60 + "\n")

total = 0

# Math (40k) - SAT Math only
print("Math (40k):")
total += quick_export(
    'ndavidson/sat-math-chain-of-thought', 'train',
    'tunix_data_preprocessed/sat_math_cot.jsonl', 40000,
    {'problem': ['question'], 'reasoning': ['reasoning_chain'], 'answer': ['answer']}
)

# Coding (15k) - Codeforces only  
print("Coding (15k):")
total += quick_export(
    'open-r1/codeforces-cots', 'train',
    'tunix_data_preprocessed/codeforces_cots.jsonl', 15000,
    {'problem': ['title'], 'reasoning': ['messages'], 'answer': ['editorial']}
)

# Science (8k) - ARC only
print("Science (8k):")
total += quick_export(
    'Locutusque/arc-cot', 'train',
    'tunix_data_preprocessed/arc_cot.jsonl', 8000,
    {'problem': ['question'], 'reasoning': ['explanation'], 'answer': ['answer']}
)

# Turing (8k)
print("Turing Reason (8k):")
total += quick_export(
    'prithivmlmods/Turing-Reason-CoT', 'train',
    'tunix_data_preprocessed/turing_reason_cot.jsonl', 8000,
    {'problem': ['question'], 'reasoning': ['chain_of_thought'], 'answer': ['solution']}
)

# Isaiah (7k)
print("Isaiah CoT (7k):")
total += quick_export(
    'isaiahbjork/chain-of-thought', 'train',
    'tunix_data_preprocessed/isaiah_chain_of_thought.jsonl', 7000,
    {'problem': ['prompt'], 'reasoning': ['reasoning'], 'answer': ['completion']}
)

# Demeter (10k)
print("Demeter LongCoT (10k):")
total += quick_export(
    'prithivmlmods/Demeter-LongCoT-6M', 'train',
    'tunix_data_preprocessed/demeter_longcot.jsonl', 10000,
    {'problem': ['instruction'], 'reasoning': ['long_reasoning'], 'answer': ['response']}
)

# Atlas (10k)
print("Atlas-Think-CoT (10k):")
total += quick_export(
    'prithivMLmods/Atlas-Think-CoT-12M', 'train',
    'tunix_data_preprocessed/atlas_think_cot.jsonl', 10000,
    {'problem': ['prompt'], 'reasoning': ['reasoning'], 'answer': ['response']}
)

print("\n" + "="*60)
print(f"TOTAL: {total:,} examples")
print("="*60 + "\n")

# Verify files
print("Output Files:")
for fname in sorted(os.listdir("tunix_data_preprocessed")):
    if fname.endswith(".jsonl"):
        fpath = os.path.join("tunix_data_preprocessed", fname)
        lines = len(open(fpath).readlines())
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"  ✓ {fname}: {lines:,} examples ({size_mb:.1f} MB)")

print(f"\n✓ Preprocessing complete! Check tunix_data_preprocessed/")
