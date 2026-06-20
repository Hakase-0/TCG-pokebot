"""
add_deck.py — import a deck straight from a LimitlessTCG decklist URL.

LimitlessTCG renders each card as <a href=".../cards/SET/NUM">N Name (SET-NUM)</a>.
We pull those, hand them to import_deck.import_decklist (which maps to engine card
IDs with energy-by-type / exact-printing / name fallback), and save an engine deck.

  python add_deck.py https://play.limitlesstcg.com/tournament/<id>/player/<p>/decklist \\
      --name dragapult --csv EN_Card_Data.csv

Used by build_deck_pool.py to pull the whole metagame; also handy standalone for
your teammate to add any list by pasting its URL.
"""
from __future__ import annotations
import argparse, os, re, sys, urllib.request

sys.path.insert(0, ".")
from import_deck import import_decklist, write_deck_csv

_UA = {"User-Agent": "Mozilla/5.0 (compatible; tcg-pokebot/1.0)"}


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def parse_decklist_html(html):
    """Return decklist text ('<count> <name> <SET> <num>' per line) from a Limitless page."""
    lines = []
    for m in re.finditer(r'cards/([A-Z]+)/(\d+)[A-Za-z]?[^>]*>\s*([^<]+?)\s*</a>', html):
        st, num, text = m.group(1), m.group(2), m.group(3).strip()
        mm = re.match(r'^(\d+)\s+(.*)$', text)
        if not mm:
            continue
        name = re.sub(r'\s*\([A-Z]+-\d+[A-Za-z]?\)\s*$', '', mm.group(2)).strip()
        lines.append(f"{mm.group(1)} {name} {st} {num}")
    return "\n".join(lines)


def import_url(url, csv_path):
    text = parse_decklist_html(fetch(url))
    ids, rep = import_decklist(text, csv_path)
    return ids, rep, text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--name", required=True, help="deck name -> decks/<name>.txt/.csv")
    ap.add_argument("--csv", default="EN_Card_Data.csv")
    ap.add_argument("--outdir", default="decks")
    a = ap.parse_args()
    ids, rep, text = import_url(a.url, a.csv)
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, a.name + ".txt"), "w").write(text + "\n")
    print(f"{a.name}: {rep['mapped']} mapped, legal60={rep['legal_60']}",
          "" if not rep['unmatched'] else f"UNMATCHED {rep['unmatched']}")
    if rep["legal_60"]:
        write_deck_csv(ids, os.path.join(a.outdir, a.name + ".csv"))
