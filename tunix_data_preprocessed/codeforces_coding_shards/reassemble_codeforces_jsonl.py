#!/usr/bin/env python3
from pathlib import Path

def main():
    shards_dir = Path(__file__).parent
    out_path = shards_dir.parent / "codeforces_coding.jsonl"
    parts = sorted(shards_dir.glob("codeforces_coding.jsonl.part-*"))
    if not parts:
        print("No parts found.")
        return
    with out_path.open('wb') as out:
        for p in parts:
            out.write(p.read_bytes())
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
