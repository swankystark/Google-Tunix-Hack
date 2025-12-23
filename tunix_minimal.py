# Minimal Tunix Preprocessing - Reliable Datasets Only
from datasets import load_dataset
import json, os

os.makedirs("tunix_data_preprocessed", exist_ok=True)

def format_tunix(problem, reasoning, answer):
    return (f"<start_of_turn>user\n{str(problem).strip()[:1500]}\n<end_of_turn>\n"
            f"<start_of_turn>model\n<reasoning>\n{str(reasoning).strip()[:3000]}\n</reasoning>\n"
            f"<answer>\n{str(answer).strip()[:500]}\n</answer>\n<end_of_turn>")

def process(name, dataset_id, count, outfile, extractors):
    """Process a single dataset."""
    try:
        print(f"{name}...", end=" ", flush=True)
        ds = load_dataset(dataset_id, split='train')
        written = 0
        
        for i, ex in enumerate(ds.shuffle(seed=42)):
            if i >= count * 3:  # Oversample
                break
            try:
                # Extract fields based on dict of possible keys
                problem = ""
                for key in extractors.get('problem', []):
                    if key in ex and ex[key]:
                        problem = str(ex[key])[:1500]
                        break
                        
                reasoning = ""
                for key in extractors.get('reasoning', []):
                    if key in ex and ex[key]:
                        val = ex[key]
                        if isinstance(val, dict) and 'steps' in val:
                            # Handle SAT Math nested format
                            reasoning = " ".join([s.get('explanation', '') for s in val.get('steps', []) if isinstance(s, dict)])
                        elif isinstance(val, list):
                            reasoning = " ".join([str(v) for v in val])
                        else:
                            reasoning = str(val)
                        break
                
                answer = ""
                for key in extractors.get('answer', []):
                    if key in ex and ex[key]:
                        answer = str(ex[key])[:500]
                        break
                
                if len(reasoning.strip()) < 30:
                    continue
                
                text = format_tunix(problem, reasoning, answer)
                with open(outfile, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1
                if written >= count:
                    break
            except:
                continue
        
        print(f"OK {written}")
        return written
    except Exception as e:
        print(f"FAIL {str(e)[:50]}")
        return 0

# Clear old files
for f in os.listdir("tunix_data_preprocessed"):
    if f.endswith(".jsonl"):
        try:
            os.remove(os.path.join("tunix_data_preprocessed", f))
        except:
            pass

print("\n" + "="*60)
print("TUNIX MINIMAL PREPROCESSING")
print("="*60 + "\n")

total = 0

# SAT Math (40k) - WORKS WELL
total += process(
    "SAT Math", 'ndavidson/sat-math-chain-of-thought', 40000,
    'tunix_data_preprocessed/sat_math_cot.jsonl',
    {'problem': ['question'], 'reasoning': ['reasoning_chain'], 'answer': ['answer']}
)

# ARC (8k) - SIMPLE
total += process(
    "ARC Science", 'Locutusque/arc-cot', 8000,
    'tunix_data_preprocessed/arc_cot.jsonl',
    {'problem': ['question', 'stem'], 'reasoning': ['explanation', 'reasoning'], 'answer': ['answer', 'label']}
)

# Turing (8k) - SIMPLE
total += process(
    "Turing Reason", 'prithivmlmods/Turing-Reason-CoT', 8000,
    'tunix_data_preprocessed/turing_reason_cot.jsonl',
    {'problem': ['question', 'input'], 'reasoning': ['chain_of_thought', 'reasoning'], 'answer': ['solution', 'answer']}
)

# Isaiah (7k) - SIMPLE
total += process(
    "Isaiah CoT", 'isaiahbjork/chain-of-thought', 7000,
    'tunix_data_preprocessed/isaiah_chain_of_thought.jsonl',
    {'problem': ['prompt', 'input'], 'reasoning': ['reasoning'], 'answer': ['completion']}
)

# Demeter (10k) - LARGE
total += process(
    "Demeter LongCoT", 'prithivmlmods/Demeter-LongCoT-6M', 10000,
    'tunix_data_preprocessed/demeter_longcot.jsonl',
    {'problem': ['instruction', 'prompt'], 'reasoning': ['long_reasoning', 'reasoning'], 'answer': ['response', 'answer']}
)

# Atlas (10k) - LARGE
total += process(
    "Atlas-Think-CoT", 'prithivMLmods/Atlas-Think-CoT-12M', 10000,
    'tunix_data_preprocessed/atlas_think_cot.jsonl',
    {'problem': ['prompt', 'instruction'], 'reasoning': ['reasoning', 'chain_of_thought'], 'answer': ['response']}
)

print("\n" + "="*60)
print(f"TOTAL: {total:,} / 93,000 target (skipped Codeforces)")
print("="*60 + "\n")

# Verify and report
print("Output Files:\n")
for fname in sorted(os.listdir("tunix_data_preprocessed")):
    if fname.endswith(".jsonl"):
        fpath = os.path.join("tunix_data_preprocessed", fname)
        try:
            lines = len(open(fpath).readlines())
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"  OK {fname}")
            print(f"    {lines:,} examples | {size_mb:.1f} MB")
        except:
            print(f"  FAIL {fname} - error reading")

print(f"\nOK Preprocessing complete!")
