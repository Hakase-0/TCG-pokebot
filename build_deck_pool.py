"""
build_deck_pool.py — pull the whole POR-rotation metagame into decks/.

Reads the Limitless metagame ranking, and for each archetype grabs the top
finisher's decklist, imports it to an engine deck (import_deck), and saves
decks/<slug>.txt/.csv. Decks that don't import to a legal 60 (e.g. they include
post-POR cards outside the engine pool) are reported and skipped.

  python build_deck_pool.py --top 30 --csv EN_Card_Data.csv

Network required (Kaggle has it; the repo Dockerfile too). Re-run to refresh.
"""
from __future__ import annotations
import argparse, os, re, sys, time

sys.path.insert(0, ".")
from add_deck import fetch, parse_decklist_html
from import_deck import import_decklist, write_deck_csv

META = "https://play.limitlesstcg.com/decks?format=standard&rotation=2026&set=POR"
ARCH = "https://play.limitlesstcg.com/decks/{slug}?format=standard&rotation=2026&set=POR"


def metagame_slugs(limit=None):
    html = fetch(META)
    seen, slugs = set(), []
    for m in re.finditer(r'/decks/([a-z0-9-]+)\?format=standard', html):
        s = m.group(1)
        if s in seen or s == "other":
            continue
        seen.add(s); slugs.append(s)
    return slugs[:limit] if limit else slugs


def first_decklist_url(slug):
    html = fetch(ARCH.format(slug=slug))
    m = re.search(r'(/tournament/[a-f0-9]+/player/[^/"]+/decklist)', html)
    return "https://play.limitlesstcg.com" + m.group(1) if m else None


def decklist_urls(slug, n):
    """Up to n distinct decklist URLs from one archetype page (e.g. the 'other' bucket)."""
    html = fetch(ARCH.format(slug=slug))
    seen, out = set(), []
    for m in re.finditer(r'/tournament/[a-f0-9]+/player/[^/"]+/decklist', html):
        u = m.group(0)
        if u not in seen:
            seen.add(u); out.append("https://play.limitlesstcg.com" + u)
        if len(out) >= n:
            break
    return out


def build_from_archetype(slug, n, csv_path, outdir, prefix, sleep=0.6):
    """Pull n decklists from a single archetype slug (used for off-meta adversaries)."""
    os.makedirs(outdir, exist_ok=True)
    urls = decklist_urls(slug, n)
    print(f"{len(urls)} decklists from '{slug}'")
    ok = 0
    for i, url in enumerate(urls, 1):
        try:
            text = parse_decklist_html(fetch(url))
            ids, rep = import_decklist(text, csv_path)
            name = f"{prefix}-{i}"
            if rep["legal_60"]:
                open(os.path.join(outdir, name + ".txt"), "w").write(text + "\n")
                write_deck_csv(ids, os.path.join(outdir, name + ".csv"))
                ok += 1
                print(f"  {name}: OK ({rep['mapped']} mapped)")
            else:
                print(f"  {name}: SKIP (unmatched={rep['unmatched'][:2]})")
        except Exception as e:
            print(f"  [{i}] ERROR {str(e)[:60]}")
        time.sleep(sleep)
    print(f"{ok} decks -> {outdir}/")
    return ok


def build(top, csv_path, outdir, sleep=0.6):
    os.makedirs(outdir, exist_ok=True)
    slugs = metagame_slugs(top)
    print(f"{len(slugs)} archetypes from the POR metagame")
    ok = skipped = 0
    for i, slug in enumerate(slugs, 1):
        try:
            url = first_decklist_url(slug)
            if not url:
                print(f"  [{i:2}] {slug:28} no decklist link"); skipped += 1; continue
            text = parse_decklist_html(fetch(url))
            ids, rep = import_decklist(text, csv_path)
            tag = f"{rep['mapped']:>2} mapped"
            if rep["legal_60"]:
                open(os.path.join(outdir, slug + ".txt"), "w").write(text + "\n")
                write_deck_csv(ids, os.path.join(outdir, slug + ".csv"))
                ok += 1
                print(f"  [{i:2}] {slug:28} OK ({tag})"
                      + (f"  ~{len(rep['unmatched'])} unmatched" if rep['unmatched'] else ""))
            else:
                skipped += 1
                print(f"  [{i:2}] {slug:28} SKIP ({tag}, not legal-60; "
                      f"unmatched={rep['unmatched'][:2]})")
        except Exception as e:
            skipped += 1
            print(f"  [{i:2}] {slug:28} ERROR {str(e)[:60]}")
        time.sleep(sleep)
    print(f"\npool: {ok} decks imported, {skipped} skipped -> {outdir}/")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30, help="how many top archetypes (0 = all)")
    ap.add_argument("--csv", default="EN_Card_Data.csv")
    ap.add_argument("--outdir", default="decks")
    ap.add_argument("--archetype", default=None,
                    help="pull N decklists from one slug (e.g. 'other' for off-meta adversaries)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prefix", default="adv")
    a = ap.parse_args()
    if a.archetype:
        build_from_archetype(a.archetype, a.n, a.csv, a.outdir, a.prefix)
    else:
        build(a.top or None, a.csv, a.outdir)
