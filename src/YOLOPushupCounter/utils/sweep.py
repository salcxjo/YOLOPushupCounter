import itertools

import counter
from evaluate import load_angles, load_truth, OUT_DIR, IMGSZ, TRUTH_FILE

# Clips used only to CHECK the winner, never to pick it. Change to taste.
HOLDOUT = {"20261104_134530", "20261104_141058"}

# Narrowed around last round's winner. UP and SMOOTH sat on the edge of the old
# grid, so those ranges extend further in that direction. MIN_REP_GAP_S is new.
GRID = {
    "DOWN_ANGLE": [115, 120, 125],
    "UP_ANGLE": [135, 140, 145],
    "DWELL_S": [0.0, 0.05],
    "SMOOTH_S": [0.07, 0.10],
    "MIN_REP_GAP_S": [0.3, 0.5],
}
KEYS = list(GRID)
ORIGINAL = {k: getattr(counter, k) for k in KEYS}   # whatever counter.py ships with

# Hand-picked settings to compare on every clip.
CANDIDATES = {
    "original": {"DOWN_ANGLE": 110, "UP_ANGLE": 150, "DWELL_S": 0.10,
                 "SMOOTH_S": 0.15, "MIN_REP_GAP_S": 0.5},
    "pick-120": {"DOWN_ANGLE": 120, "UP_ANGLE": 140, "DWELL_S": 0.05,
                 "SMOOTH_S": 0.10, "MIN_REP_GAP_S": 0.5},
    "pick-125": {"DOWN_ANGLE": 125, "UP_ANGLE": 140, "DWELL_S": 0.05,
                 "SMOOTH_S": 0.10, "MIN_REP_GAP_S": 0.5},
}
DEPTH_REPORT_FOR = "pick-120"   # which candidate to print rep depths for


def cost(diff):
    """Phantom reps (diff > 0) count double: a leaderboard needs trustworthy counts."""
    return 2 * diff if diff > 0 else -diff


def run_all(params, clips):
    """Run the counter on every clip under `params`; always restore the defaults."""
    for k in KEYS:
        setattr(counter, k, params.get(k, ORIGINAL[k]))
    try:
        return {s: counter.count_reps(d["t_s"], d["elbow_left"], d["elbow_right"],
                                      d["torso_tilt"]) for s, d in clips.items()}
    finally:
        for k, v in ORIGINAL.items():
            setattr(counter, k, v)


def diffs_of(results, truth):
    return {s: r["count"] - truth[s][0] for s, r in results.items()}


def show(p):
    return (f"DOWN {p['DOWN_ANGLE']} UP {p['UP_ANGLE']} dwell {p['DWELL_S']} "
            f"smooth {p['SMOOTH_S']} gap {p['MIN_REP_GAP_S']}")


def main():
    truth = load_truth(TRUTH_FILE)
    clips = {}
    for stem in truth:
        p = OUT_DIR / f"{stem}_{IMGSZ}" / "angles.csv"
        if p.exists():
            clips[stem] = load_angles(p)
    tune = {s: d for s, d in clips.items() if s not in HOLDOUT}
    held = {s: d for s, d in clips.items() if s in HOLDOUT}
    ordered = {**tune, **held}
    print(f"tuning clips: {len(tune)}   held-out clips: {len(held)} (marked *)\n")

    # 1) grid search, scored on the tuning clips only
    rows = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(KEYS, combo))
        diffs = diffs_of(run_all(params, tune), truth)
        rows.append((sum(cost(x) for x in diffs.values()), params, diffs))
    rows.sort(key=lambda r: r[0])

    print("BEST GRID COMBINATIONS (diff = counted - true, tuning clips only)")
    for c, params, diffs in rows[:8]:
        print(f"  cost {c:>3}  {show(params)}   {' '.join(f'{v:+d}' for v in diffs.values())}")

    # 2) hand-picked candidates on every clip
    print("\nCANDIDATES, diff per clip")
    print(f"  {'':<10}" + "".join(f"{(s[-6:] + ('*' if s in held else '')):>8}" for s in ordered)
          + f"{'tune cost':>11}{'held cost':>11}")
    all_results = {}
    for label, params in CANDIDATES.items():
        res = run_all(params, ordered)
        all_results[label] = res
        diffs = diffs_of(res, truth)
        tc = sum(cost(v) for s, v in diffs.items() if s in tune)
        hc = sum(cost(v) for s, v in diffs.items() if s in held)
        print(f"  {label:<10}" + "".join(f"{v:>+8d}" for v in diffs.values())
              + f"{tc:>11}{hc:>11}")

    # 3) rep depths for every clip the chosen candidate gets wrong
    res = all_results.get(DEPTH_REPORT_FOR)
    if res:
        print(f"\nREP DEPTHS for clips where '{DEPTH_REPORT_FOR}' is wrong "
              "(deepest angle of each counted rep)")
        any_wrong = False
        for stem, r in res.items():
            diff = r["count"] - truth[stem][0]
            if diff != 0:
                any_wrong = True
                depths = [round(x) for x in r["rep_depths"]]
                print(f"  {stem}  true {truth[stem][0]}  counted {r['count']}  depths {depths}")
        if not any_wrong:
            print("  none, every clip is exact")


if __name__ == "__main__":
    main()