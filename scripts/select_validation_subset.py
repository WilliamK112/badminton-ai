#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a subset of validation rows by row index")
    p.add_argument("--input", required=True, help="Input JSONL path")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--indices", required=True, help="Comma-separated zero-based row indices, e.g. 0,1,7,8")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    wanted = {int(x.strip()) for x in args.indices.split(',') if x.strip()}
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    picked = [row for idx, row in enumerate(rows) if idx in wanted]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fp:
        for row in picked:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"selected {len(picked)} / {len(rows)} rows -> {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

