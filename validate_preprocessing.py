import json
import os

print("\n" + "="*70)
print("PREPROCESSED DATASET VALIDATION")
print("="*70 + "\n")

base_dir = "tunix_data_preprocessed"
total_examples = 0
all_files_ok = True

for fname in sorted(os.listdir(base_dir)):
    if fname.endswith(".jsonl"):
        fpath = os.path.join(base_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = 0
                valid = 0
                invalid = 0
                
                for line in f:
                    lines += 1
                    try:
                        data = json.loads(line)
                        if 'text' in data and len(data['text']) > 100:
                            valid += 1
                        else:
                            invalid += 1
                    except:
                        invalid += 1
                
                size_mb = os.path.getsize(fpath) / (1024*1024)
                total_examples += valid
                
                status = "OK" if invalid == 0 else f"WARN ({invalid} invalid)"
                print(f"[{status:20s}] {fname:35s} {valid:8,d} valid")
                
                if invalid > 0:
                    all_files_ok = False
                    
        except Exception as e:
            print(f"[ERROR               ] {fname:35s} {str(e)[:40]}")
            all_files_ok = False

print("\n" + "="*70)
print(f"TOTAL VALID EXAMPLES: {total_examples:,}")
print(f"STATUS: {'ALL FILES VALID' if all_files_ok else 'SOME FILES HAVE ISSUES'}")
print("="*70 + "\n")

if total_examples > 100000:
    print(f"SUCCESS! Dataset contains {total_examples:,} valid training examples")
    print(f"Target was ~125,000 and we have {total_examples:,} ({100*total_examples//125000}%)")
