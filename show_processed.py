import json
from datasets import load_from_disk

ds = load_from_disk('processed_arc_cot')

for i in range(min(5, len(ds))):
    item = ds[i]
    # Convert any non-serializable elements
    print(json.dumps(item, ensure_ascii=False, indent=2))

print(f"\nTotal examples in disk dataset: {len(ds)}")
