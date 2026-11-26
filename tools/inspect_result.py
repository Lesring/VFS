"""查看某次结果的槽位构成，调试用。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path or not path.exists():
        root = Path("data/outputs")
        cands = sorted(root.glob("*/result.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            print("no results")
            return 1
        path = cands[-1]
    d = json.loads(path.read_text(encoding="utf-8"))
    ev = {e["id"]: e for e in d.get("evidence", [])}
    only = sys.argv[2:] if len(sys.argv) > 2 else None
    for slot, s in (d.get("slots") or {}).items():
        if only and slot not in only:
            continue
        print(f"\n=== {slot} / {s['label']} conf={s['confidence']} ({s['confidence_label']})")
        for c in s.get("candidates", [])[:4]:
            print(f"  - {c['value']}  score={c['score']}  details={ {k: v for k, v in c.get('details', {}).items() if k in ('iata','icao','lat','lon','from_route','canon_key')} }")
            for eid in c.get("evidence_ids", []):
                e = ev.get(eid)
                if e:
                    print(f"      {eid} [{e['kind']}/{e['source']}] c={e['confidence']} {e['value'][:60]}")
        if s.get("notes"):
            print("  notes:", s["notes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
