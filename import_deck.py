"""
import_deck.py — turn a LimitlessTCG decklist into an engine deck (60 card IDs).

Limitless lists cards as "<count> <name> <SET>-<num>" (e.g. "3 Dragapult ex
TWM-130"). The engine identifies cards by its own Card ID, and a given card may
be present under a *different* printing than the one a list cites. So we map:

  1. basic energy ("Psychic Energy" ...) -> the engine's Basic {X} Energy by type
  2. exact (Expansion, Collection No.) match
  3. fall back to NAME match (any legal printing in the pool) — flagged as a
     substitution so you can eyeball it
  4. anything still unmatched is reported (likely a post-POR / CRI card)

Names are normalized (curly->straight apostrophe, case, spacing) because the
card data mixes encodings. Verifies the result is exactly 60 cards.

Usage:
  from import_deck import import_decklist
  ids, report = import_decklist(open("decks/dragapult.txt").read(), "EN_Card_Data.csv")
"""
from __future__ import annotations
import csv, re, os

_ENERGY_SYMBOL = {
    "grass": "G", "fire": "R", "water": "W", "lightning": "L", "psychic": "P",
    "fighting": "F", "darkness": "D", "metal": "M", "fairy": "Y",
}


def _norm(s):
    s = s.replace("\u2019", "'").replace("\u2018", "'").lower().strip()
    return re.sub(r"\s+", " ", s)


def _load_card_data(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_setnum, by_name, energy_by_sym = {}, {}, {}
    for r in rows:
        cid = r.get("Card ID", "").strip()
        if not cid.isdigit():
            continue
        cid = int(cid)
        name = r["Card Name"]
        exp = (r.get("Expansion") or "").strip()
        num = (r.get("Collection No.") or "").strip()
        by_setnum[(exp.upper(), num.lstrip("0") or "0")] = cid
        by_name.setdefault(_norm(name), cid)
        if name.startswith("Basic ") and "Energy" in name:
            m = re.search(r"\{(\w)\}", name)
            if m:
                energy_by_sym[m.group(1)] = cid
    return by_setnum, by_name, energy_by_sym


_LINE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s*(?:[\(\[]?([A-Za-z]{2,4})[-\s](\d+[A-Za-z]?)[\)\]]?)?\s*$"
)


def parse_decklist(text):
    """Yield (count, name, set_or_None, num_or_None) for each card line."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r"^(pok[eé]mon|trainer|energy)\b.*\(\d+\)\s*$", line, re.I):
            continue
        if line.lower().startswith(("pokémon", "pokemon", "trainer", "energy")) and "(" in line:
            continue
        m = _LINE.match(line)
        if not m:
            continue
        cnt, name, st, num = m.groups()
        out.append((int(cnt), name.strip(), (st or "").upper() or None, num))
    return out


def import_decklist(text, csv_path):
    by_setnum, by_name, energy_by_sym = _load_card_data(csv_path)
    ids, report = [], {"mapped": 0, "by_setnum": 0, "by_name": 0, "energy": 0,
                       "substitutions": [], "unmatched": [], "total_cards": 0}
    for cnt, name, st, num in parse_decklist(text):
        nm = _norm(name)
        cid = None
        # 1. basic energy
        base = nm.replace(" energy", "").strip()
        if base in _ENERGY_SYMBOL and ("energy" in nm):
            cid = energy_by_sym.get(_ENERGY_SYMBOL[base])
            if cid:
                report["energy"] += 1
        # 2. exact printing
        if cid is None and st and num is not None:
            cid = by_setnum.get((st, num.lstrip("0") or "0"))
            if cid:
                report["by_setnum"] += 1
        # 3. name fallback
        if cid is None:
            cid = by_name.get(nm)
            if cid:
                report["by_name"] += 1
                report["substitutions"].append(f"{name} {st or ''}-{num or ''} -> id {cid} (by name)")
        if cid is None:
            report["unmatched"].append(f"{cnt} {name} {st or ''}-{num or ''}")
            continue
        ids.extend([cid] * cnt)
        report["mapped"] += cnt
    report["total_cards"] = len(ids)
    report["legal_60"] = (len(ids) == 60)
    return ids, report


def write_deck_csv(ids, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(str(i) for i in ids))


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="decklist text file (Limitless format)")
    ap.add_argument("--csv", default="EN_Card_Data.csv")
    ap.add_argument("--out", default=None, help="write engine deck.csv here")
    a = ap.parse_args()
    ids, rep = import_decklist(open(a.infile).read(), a.csv)
    print(json.dumps({k: v for k, v in rep.items() if k != "substitutions"}, indent=2))
    if rep["substitutions"]:
        print("substitutions:")
        for s in rep["substitutions"]:
            print("  ", s)
    if a.out and rep["legal_60"]:
        write_deck_csv(ids, a.out)
        print(f"wrote {a.out}")
