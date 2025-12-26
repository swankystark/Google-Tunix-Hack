import json

path = 'processed_arc_cot_formatted.jsonl'
print('Samples from', path)
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        rec = json.loads(line)
        print(f"{i}:\n{rec['input']}\n{rec['output']}\n")
