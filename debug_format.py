import json

fname = "tunix_data_preprocessed/category_creative_ideation.jsonl"
print(f"Checking {fname}")

with open(fname, 'rb') as f:
    first_bytes = f.read(300)
    print(f"First 300 bytes (raw): {first_bytes[:100]}")

print("\nTrying to parse first line:")
with open(fname) as f:
    line = f.readline()
    print(f"Line type: {type(line)}")
    print(f"Line length: {len(line)}")
    print(f"First 100 chars: {line[:100]}")
    print(f"Starts with quote?: {line[0] == '\"'}")
    
    # Try to parse it
    try:
        data = json.loads(line)
        print(f"JSON parsed successfully: {type(data)}")
        if isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
    except Exception as e:
        print(f"JSON parse error: {e}")
        
        # Check if it's a string that looks like JSON
        if line.startswith('"'):
            print("\nIt's a quoted string, trying to unescape...")
            try:
                unquoted = json.loads(line)
                print(f"Unquoted: {unquoted[:100]}...")
            except:
                pass
