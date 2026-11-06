import itertools

import counter
from evaluate import load_angles, load_truth, OUT_DIR, IMGSZ, TRUTH_FILE

# Clips used only to CHECK the winner, never to pick it. Change to taste.
HOLDOUT = {"20261104_134530", "20261104_141058"}

GRID = {
    "DOWN_ANGLE": [110, 115, 120, 125],
    "UP_ANGLE": [140, 145, 150],
    "DWELL_S": [0.0, 0.05, 0.1],
    "SMOOTH_S": [0.10, 0.15],
}
KEYS = list(GRID)
ORIGINAL = {k: getattr(counter, k) for k in KEYS}


def cost(diff):
    """Phantom reps (diff > 0) count double: a leaderboard needs trustworthy counts."""
    return 2 * diff if diff > 0 else -diff


def diffs_for(params, clips, truth):
    for k, v in params.items():
        setattr(counter, k, v)
    out = {}
    for stem, d in clips.items():
        res = counter.count_reps(d["t_s"], d["elbow_left"], d["elbow_right"], d["torso_tilt"])
        out[stem] = res["count"] - truth[stem][0]
    for k, v in ORIGINAL.items():          # always restore
        setattr(counter, k, v)
    return out


def fmt(diffs):
    return " ".join(f"{v:+d}" for v in diffs.values())


def main():
    truth = load_truth(TRUTH_FILE)
    clips = {}
    for stem in truth:
        p = OUT_DIR / f"{stem}_{IMGSZ}" / "angles.csv"
        if p.exists():
            clips[stem] = load_angles(p)
    tune = {s: d for s, d in clips.items() if s not in HOLDOUT}
    held = {s: d for s, d in clips.items() if s in HOLDOUT}
    print(f"tuning clips: {len(tune)}   held-out clips: {len(held)}\n")

    rows = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(KEYS, combo))
        diffs = diffs_for(params, tune, truth)
        rows.append((sum(cost(x) for x in diffs.values()), params, diffs))
    rows.sort(key=lambda r: r[0])

    base = diffs_for(ORIGINAL, tune, truth)
    print(f"current settings  cost {sum(cost(x) for x in base.values()):>3}   diffs: {fmt(base)}")
    print("best combinations (diff per tuning clip, counted minus true):")
    for c, params, diffs in rows[:8]:
        print(f"  cost {c:>3}  {params}  diffs: {fmt(diffs)}")

    CANDIDATES = [
        {"DOWN_ANGLE": 120, "UP_ANGLE": 140, "DWELL_S": 0.05, "SMOOTH_S": 0.10},
        {"DOWN_ANGLE": 125, "UP_ANGLE": 140, "DWELL_S": 0.05, "SMOOTH_S": 0.10},
    ]
    # ...at the end of main(), replacing the HELD-OUT block:
    print("\nCANDIDATES (tuning diffs | held-out diffs)")
    for params in CANDIDATES:
        print(f"  {params}")
        print(f"     {fmt(diffs_for(params, tune, truth))} | {fmt(diffs_for(params, held, truth))}")


if __name__ == "__main__":
    main()