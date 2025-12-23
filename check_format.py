import json

fname = "tunix_data_preprocessed/category_creative_ideation.jsonl"
with open(fname) as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            if isinstance(data, dict) and 'text' in data:
                text = data['text']
                print(f"Line {i}: OK - format valid, text length {len(text)}")
                if len(text) < 100:
                    print(f"  WARNING: text too short ({len(text)} chars)")
            else:
                print(f"Line {i}: INVALID - not a dict or missing text key")
                print(f"  Type: {type(data)}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
            if i >= 2:
                break
        except Exception as e:
            print(f"Line {i}: JSON ERROR - {e}")
            break
