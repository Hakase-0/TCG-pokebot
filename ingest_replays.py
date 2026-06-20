"""
ingest_replays.py — behavioral-cloning data from REAL games (replay files).

A Kaggle episode replay is kaggle-environments JSON: `steps` is a list of
per-step, per-agent states, each `{observation, action, reward, status}`. For a
turn-based game the agent that acted on a step has both an encodable
`observation` (the obs_dict with `select`/`current`/`logs`) and the `action`
(the option indices it returned) — which is exactly a BC sample.

This produces the SAME .pkl `train_bc.py` consumes, so cloning from real games
is a drop-in swap for `gen_selfplay_data.py`. Key quality lever: `--winners-only`
keeps just the winning side's decisions, so you imitate good play, not bad.

The exact cabt observation nesting can't be verified offline (the env plugin
only exists on Kaggle), so the parser is tolerant — it finds any obs_dict +
action pair — and `--inspect` dumps an unknown replay's structure so the field
mapping can be adjusted in one line if needed.

Getting replays: from the competition's "My Submissions", each game has a
downloadable JSON replay; or pull them with the Kaggle API. Drop them in
`replays/` and run this.

Usage:
  python ingest_replays.py --replays replays/ --winners-only --out data/bc.pkl
  python ingest_replays.py --inspect replays/somegame.json
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import features


def _looks_like_obs(o):
    return isinstance(o, dict) and "select" in o and "current" in o


def _looks_like_action(a):
    return isinstance(a, list) and a and all(isinstance(x, int) for x in a)


def _winner(j):
    """Winning agent index from final rewards (or statuses), else None."""
    if not isinstance(j, dict):
        return None
    rewards = j.get("rewards")
    if isinstance(rewards, list) and len(rewards) == 2 and None not in rewards:
        if rewards[0] != rewards[1]:
            return 0 if rewards[0] > rewards[1] else 1
    steps = j.get("steps")
    if steps and isinstance(steps[-1], list):
        rs = [s.get("reward") for s in steps[-1] if isinstance(s, dict)]
        if len(rs) == 2 and None not in rs and rs[0] != rs[1]:
            return 0 if rs[0] > rs[1] else 1
    return None


def iter_decisions(j):
    """Yield (obs_dict, action, player_index, winner_index) for each decision."""
    winner = _winner(j)
    steps = j.get("steps") if isinstance(j, dict) else (j if isinstance(j, list) else None)
    if steps and isinstance(steps[0], list):                 # standard schema
        for stepstates in steps:
            for i, st in enumerate(stepstates):
                if not isinstance(st, dict):
                    continue
                obs, act = st.get("observation"), st.get("action")
                if _looks_like_obs(obs) and _looks_like_action(act) and obs.get("select") is not None:
                    yield obs, act, i, winner
        return
    records = j if isinstance(j, list) else (j.get("records", []) if isinstance(j, dict) else [])
    for r in records:                                        # tolerant flat records
        if not isinstance(r, dict):
            continue
        obs = r.get("observation") or r.get("obs")
        act = r.get("action")
        if _looks_like_obs(obs) and _looks_like_action(act) and obs.get("select") is not None:
            yield obs, act, r.get("player"), r.get("winner", winner)


def build(paths, out, attack_db=None, winners_only=False, player=None, log=None):
    samples = files = kept = skipped = 0
    samples = []
    for path in paths:
        try:
            j = json.load(open(path))
        except Exception as e:
            print("  skip", path, "-", e); continue
        files += 1
        for obs, act, pidx, winner in iter_decisions(j):
            if player is not None and pidx is not None and pidx != player:
                continue
            if winners_only and winner is not None and pidx is not None and pidx != winner:
                skipped += 1; continue
            nopt = len(obs["select"].get("option", []))
            if not act or any(a >= nopt for a in act):       # skip deck-selection / malformed
                continue
            enc = features.encode_observation(obs, attack_lookup=attack_db)
            if enc is None:
                continue
            enc["target"] = list(act)
            samples.append(enc); kept += 1
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    print(f"{files} replays -> {kept} samples"
          + (f" ({skipped} dropped: not winner)" if winners_only else "")
          + f" -> {out}")
    if log:
        try:
            import stats
            stats.log(log, event="ingest", files=files, samples=kept,
                      skipped=skipped, winners_only=winners_only)
        except Exception:
            pass
    return kept


def inspect(path):
    j = json.load(open(path))
    print("top-level type:", type(j).__name__)
    if isinstance(j, dict):
        print("top-level keys:", list(j.keys()))
        print("winner (from rewards/statuses):", _winner(j))
        steps = j.get("steps")
        if steps:
            print("steps:", len(steps), "| step[0] is list of", len(steps[0]), "agents")
            st = steps[min(1, len(steps) - 1)][0]
            if isinstance(st, dict):
                print("agent-state keys:", list(st.keys()))
                obs = st.get("observation")
                print("observation is obs_dict:", _looks_like_obs(obs),
                      "| keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__)
    n = sum(1 for _ in iter_decisions(j))
    print("decisions the parser can extract:", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="replays/", help="dir or glob of replay JSON")
    ap.add_argument("--out", default="data/bc_replays.pkl")
    ap.add_argument("--winners-only", action="store_true", help="keep only the winner's moves")
    ap.add_argument("--player", type=int, default=None, help="keep only this agent index")
    ap.add_argument("--log", default="logs/ingest.jsonl")
    ap.add_argument("--inspect", default=None, help="print one replay's structure and exit")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.inspect); sys.exit(0)

    attack_db = None
    if os.path.exists("attack_table.json"):
        attack_db = {int(k): v for k, v in json.load(open("attack_table.json")).items()}
    paths = sorted(glob.glob(a.replays)) if any(c in a.replays for c in "*?[") \
        else sorted(glob.glob(os.path.join(a.replays, "*.json")))
    if not paths:
        print(f"no replay JSON found at {a.replays}"); sys.exit(1)
    build(paths, a.out, attack_db, a.winners_only, a.player, a.log)
