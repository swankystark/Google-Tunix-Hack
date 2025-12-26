import json
from datasets import load_from_disk

ds = load_from_disk('processed_arc_cot')
out_path = 'processed_arc_cot.jsonl'
count = 0
with open(out_path, 'w', encoding='utf-8') as f:
    for item in ds:
        # ensure JSON-serializable
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
        count += 1

print(f"Wrote {count} records to {out_path}")

# print first 5 lines as sample
print('\nSample (first 5 records):')
with open(out_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line.strip())
