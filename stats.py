"""stats.py — append-only JSONL logging so a terminal UI can tail training/eval stats."""
from __future__ import annotations
import json, os, time


def log(path, **record):
    record.setdefault("ts", round(time.time(), 3))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read(path):
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out
