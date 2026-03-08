import json
from pathlib import Path

sessions_dir = Path("output/sessions")
skipped = []

for fp in sorted(sessions_dir.glob("*.json")):
    try:
        doc = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        skipped.append((fp.name, f"parse error: {e}"))
        continue

    flags = doc.get("validation_flags", [])
    if "REDIRECT_STUB" in flags:
        skipped.append((fp.name, f"REDIRECT_STUB — flags={flags}"))
        continue
    if doc.get("word_count", 0) == 0:
        skipped.append((fp.name, "word_count=0"))
        continue
    turns = doc.get("turns", [])
    if not turns:
        skipped.append((fp.name, "no turns array"))

print(f"Files that would be skipped: {len(skipped)}")
for name, reason in skipped:
    print(f"  {name}: {reason}")
