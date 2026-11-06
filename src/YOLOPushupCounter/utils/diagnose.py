import numpy as np

import counter  # its constants get switched off one at a time
from evaluate import load_angles, load_truth, OUT_DIR, IMGSZ, TRUTH_FILE

NAMES = ("DOWN_ANGLE", "UP_ANGLE", "SMOOTH_S", "DWELL_S", "MIN_REP_GAP_S",
         "MAX_GAP_S", "TILT_MAX")
BASE = {k: getattr(counter, k) for k in NAMES}

CONFIGS = {
    "base":     {},
    "noTilt":   {"TILT_MAX": 1e9},
    "noGap":    {"MAX_GAP_S": 1e9},
    "noDwell":  {"DWELL_S": 0.0},
    "noSmooth": {"SMOOTH_S": 0.0},
    "loose":    {"DOWN_ANGLE": 120, "UP_ANGLE": 140},
    "noGuards": {"TILT_MAX": 1e9, "MAX_GAP_S": 1e9, "DWELL_S": 0.0, "SMOOTH_S": 0.0},
}


def run(d, overrides):
    for k in NAMES:
        setattr(counter, k, overrides.get(k, BASE[k]))
    res = counter.count_reps(d["t_s"], d["elbow_left"], d["elbow_right"], d["torso_tilt"])
    for k in NAMES:                       # always restore
        setattr(counter, k, BASE[k])
    return res


def pct(a, q):
    a = a[~np.isnan(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    truth = load_truth(TRUTH_FILE)
    data = {}
    for stem in truth:
        p = OUT_DIR / f"{stem}_{IMGSZ}" / "angles.csv"
        if p.exists():
            data[stem] = load_angles(p)

    print("SIGNAL FACTS (chosen arm)")
    print(f"{'video':<18}{'true':>5}{'valid%':>8}{'p5':>6}{'p95':>6}{'tilt50':>8}{'tilt90':>8}")
    for stem, d in data.items():
        side = run(d, {})["side"]
        a = d["elbow_left"] if side == "left" else d["elbow_right"]
        valid = 100 * np.mean(~np.isnan(a))
        print(f"{stem:<18}{truth[stem][0]:>5}{valid:>8.0f}{pct(a, 5):>6.0f}{pct(a, 95):>6.0f}"
              f"{pct(d['torso_tilt'], 50):>8.0f}{pct(d['torso_tilt'], 90):>8.0f}")

    print("\nCOUNTS WITH ONE RULE SWITCHED OFF")
    print(f"{'video':<18}{'true':>5}" + "".join(f"{n:>10}" for n in CONFIGS))
    for stem, d in data.items():
        counts = [run(d, ov)["count"] for ov in CONFIGS.values()]
        print(f"{stem:<18}{truth[stem][0]:>5}" + "".join(f"{c:>10}" for c in counts))


if __name__ == "__main__":
    main()