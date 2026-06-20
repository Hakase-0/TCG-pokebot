"""
fetch_replays.py — get real game replays into replays/ for behavioral cloning.

Three sources, easiest first:

  1. COMPETITION FORUM EXPORT (best): the organizers post a daily export of the
     top-rated episodes for BC/RL/IL. Add it to your notebook as a dataset (or
     download it) — it's already kaggle-environments JSON, so you can skip this
     script and run ingest_replays.py straight on that folder.

  2. KAGGLE CLI by episode id: this script loops `kaggle competitions replay
     <id>` with rate limiting. Give it an ids file (one episode id per line):
         python fetch_replays.py --ids episode_ids.txt --out replays/

  3. META KAGGLE (find ids yourself): the "Meta Kaggle" dataset lists every
     episode. This script can read Competitions.csv + Episodes.csv to find this
     competition's episode ids (the "do replays exist?" check), then download:
         python fetch_replays.py --meta-kaggle /path/to/meta-kaggle \
             --competition pokemon-tcg-ai-battle --limit 500 --out replays/

REQUIREMENTS YOU MUST SET UP (this script can't do it for you):
  * pip install kaggle
  * put your Kaggle API token at ~/.kaggle/kaggle.json (Account -> Create New
    API Token), chmod 600. The CLI authenticates with that; this script never
    sees your credentials.
  * accept the competition rules on the website first, or replay downloads 403.

Rate limit: Kaggle allows ~60 replay requests/minute; default sleep honors it.
"""
from __future__ import annotations
import argparse, csv, glob, os, subprocess, sys, time


def _already(out):
    have = set()
    for p in glob.glob(os.path.join(out, "*.json")):
        b = os.path.basename(p)
        for tok in b.replace("-", " ").replace(".", " ").split():
            if tok.isdigit():
                have.add(tok)
    return have


def download_ids(ids, out, sleep=1.1):
    os.makedirs(out, exist_ok=True)
    have = _already(out)
    done = skipped = failed = 0
    for eid in ids:
        eid = str(eid).strip()
        if not eid.isdigit():
            continue
        if eid in have:
            skipped += 1; continue
        try:
            subprocess.run(["kaggle", "competitions", "replay", eid, "-p", out],
                           check=True, capture_output=True, text=True)
            done += 1
        except FileNotFoundError:
            sys.exit("kaggle CLI not found — `pip install kaggle` and set up ~/.kaggle/kaggle.json")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"  episode {eid} failed: {(e.stderr or '').strip()[:120]}")
        if done % 25 == 0 and done:
            print(f"  ...{done} downloaded")
        time.sleep(sleep)               # respect 60/min
    print(f"downloaded {done}, skipped {skipped} (already have), failed {failed} -> {out}")


def _col(header, *names):
    for n in names:
        if n in header:
            return header.index(n)
    raise SystemExit(f"column {names} not in CSV header: {header}")


def meta_kaggle_ids(mk_dir, competition, limit=None, report_only=False):
    comp_csv = os.path.join(mk_dir, "Competitions.csv")
    epi_csv = os.path.join(mk_dir, "Episodes.csv")
    for p in (comp_csv, epi_csv):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — add the 'Meta Kaggle' dataset (kaggle/meta-kaggle).")
    # resolve competition id from slug (or accept a numeric id)
    comp_id = competition if competition.isdigit() else None
    if comp_id is None:
        with open(comp_csv, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f); head = next(r)
            i_id = _col(head, "Id"); i_slug = _col(head, "Slug")
            for row in r:
                if row[i_slug] == competition:
                    comp_id = row[i_id]; break
    if comp_id is None:
        sys.exit(f"competition slug '{competition}' not found in Competitions.csv")
    print(f"competition id: {comp_id}")
    # stream Episodes.csv (can be large) filtering by competition
    ids = []
    with open(epi_csv, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); head = next(r)
        i_id = _col(head, "Id"); i_comp = _col(head, "CompetitionId")
        for row in r:
            if row[i_comp] == comp_id:
                ids.append(row[i_id])
    print(f"episodes found for this competition: {len(ids)}")
    if report_only:
        return ids
    if limit:
        ids = ids[-int(limit):]            # most recent N
        print(f"taking most recent {len(ids)}")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="text file of episode ids (one per line)")
    ap.add_argument("--meta-kaggle", help="path to the Meta Kaggle dataset dir")
    ap.add_argument("--competition", default="pokemon-tcg-ai-battle")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="replays/")
    ap.add_argument("--sleep", type=float, default=1.1)
    ap.add_argument("--check", action="store_true",
                    help="Meta Kaggle: only report how many episodes exist, don't download")
    a = ap.parse_args()

    if a.meta_kaggle:
        ids = meta_kaggle_ids(a.meta_kaggle, a.competition, a.limit, report_only=a.check)
        if a.check:
            sys.exit(0)
    elif a.ids:
        ids = [l for l in open(a.ids).read().split()]
    else:
        sys.exit("give --ids FILE or --meta-kaggle DIR (see header for sources)")
    download_ids(ids, a.out, a.sleep)
