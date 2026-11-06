import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from counter import count_reps  # your code

OUT_DIR = Path("output")
IMGSZ = 416                      # must match the folders the extraction script made
TRUTH_FILE = Path("ground_truth.csv")


def load_angles(path):
    cols = {"t_s": [], "elbow_left": [], "elbow_right": [], "torso_tilt": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k in cols:
                v = row[k]
                cols[k].append(float(v) if v != "" else np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def load_truth(path):
    truth = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stem = Path(row["video"].strip()).stem
            side = (row.get("side") or "").strip().lower()
            truth[stem] = (int(row["true_reps"]), side)
    return truth


def save_overlay(d, res, out_path, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    key = {"left": "elbow_left", "right": "elbow_right"}.get(res.get("side"))
    if key:
        ax.plot(d["t_s"], d[key], lw=1.5, label=res["side"])
    else:
        ax.plot(d["t_s"], d["elbow_left"], lw=1, alpha=0.6, label="left")
        ax.plot(d["t_s"], d["elbow_right"], lw=1, alpha=0.6, label="right")
    for i, rt in enumerate(res.get("rep_times", []), start=1):
        ax.axvline(rt, color="green", ls="--", alpha=0.6)
        ax.text(rt, ax.get_ylim()[1], str(i), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("elbow angle (deg)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    truth = load_truth(TRUTH_FILE)
    results = []
    print(f"{'video':<20}{'side exp/got':<16}{'true':>6}{'counted':>9}{'diff':>6}  result")
    for stem, (true_reps, true_side) in truth.items():
        folder = OUT_DIR / f"{stem}_{IMGSZ}"
        csv_path = folder / "angles.csv"
        if not csv_path.exists():
            print(f"{stem:<20}missing {csv_path}")
            continue

        d = load_angles(csv_path)
        res = count_reps(d["t_s"], d["elbow_left"], d["elbow_right"], d["torso_tilt"])
        counted = int(res["count"])
        got_side = res.get("side", "?")
        diff = counted - true_reps
        side_ok = (not true_side) or true_side == got_side
        ok = diff == 0 and side_ok

        results.append((diff, ok))
        side_txt = f"{true_side or '-'}/{got_side}"
        print(f"{stem:<20}{side_txt:<16}{true_reps:>6}{counted:>9}{diff:>+6}  "
              f"{'PASS' if ok else 'FAIL'}")
        save_overlay(d, res, folder / "counted.png", f"{stem}: counted {counted}, true {true_reps}")

    if results:
        diffs = [r[0] for r in results]
        print(f"\npassed {sum(r[1] for r in results)}/{len(results)}")
        print(f"over-counts (phantom reps): {sum(x > 0 for x in diffs)}   "
              f"under-counts (missed reps): {sum(x < 0 for x in diffs)}")
        print(f"mean absolute error: {np.mean(np.abs(diffs)):.2f} reps")


if __name__ == "__main__":
    main()