import json
from datasets import load_from_disk

ds = load_from_disk('processed_arc_cot')
out_path = 'processed_arc_cot_formatted.jsonl'
count = 0
with open(out_path, 'w', encoding='utf-8') as f:
    for item in ds:
        msgs = item.get('messages', [])
        # Expect messages: [system, user, assistant]
        system = ''
        user = ''
        assistant = ''
        if len(msgs) >= 1:
            system = msgs[0].get('content','')
        if len(msgs) >= 2:
            user = msgs[1].get('content','')
        if len(msgs) >= 3:
            assistant = msgs[2].get('content','')
        prompt = system.strip() + "\n\n" + user.strip()
        completion = assistant.strip()
        rec = {"input": prompt, "output": completion}
        json.dump(rec, f, ensure_ascii=False)
        f.write('\n')
        count += 1
print(f"Wrote {count} records to {out_path}")
