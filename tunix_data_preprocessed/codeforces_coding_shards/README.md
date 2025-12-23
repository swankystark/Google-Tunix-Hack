# Codeforces Coding JSONL (Sharded)

This folder contains the sharded dataset `codeforces_coding.jsonl` split to stay under GitHub's 100MB file limit.

- Parts: `codeforces_coding.jsonl.part-0001`, `codeforces_coding.jsonl.part-0002`, ...
- Reassemble to a single file:

PowerShell (Windows):

```powershell
Get-Content -Path .\\codeforces_coding.jsonl.part-* -Encoding Byte -ReadCount 0 | Set-Content -Path ..\\codeforces_coding.jsonl -Encoding Byte
```

Python (cross-platform):

```bash
python reassemble_codeforces_jsonl.py
```

This will create `../codeforces_coding.jsonl` next to this `codeforces_coding_shards` folder.