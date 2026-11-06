import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---- config ----
MODEL_PATH = "yolo26n-pose.pt"
IMGSZ = 416
DEVICE = 0
MIN_CONF = 0.5
MAIN_SIDE = "right"            # the side drawn bold in the plot
VIDEO_DIR = Path("train-videos")
OUT_DIR = Path("output")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# COCO keypoints: shoulder, elbow, wrist, hip
KP = {"left": (5, 7, 9, 11), "right": (6, 8, 10, 12)}


def angle_at(a, b, c):
    """Angle in degrees at b, formed by a-b-c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))))


def tilt_from_horizontal(p, q):
    """Angle (0-90) of the line p-q relative to horizontal."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    deg = abs(np.degrees(np.arctan2(dy, dx)))
    return float(min(deg, 180 - deg))


def measure(result, side):
    """(elbow_angle, torso_tilt) for the largest person; NaN where unusable."""
    kps = result.keypoints
    if kps is None or kps.conf is None or kps.xy.shape[0] == 0:
        return np.nan, np.nan
    b = result.boxes.xyxy
    i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())
    xy = kps.xy[i].cpu().numpy()
    conf = kps.conf[i].cpu().numpy()
    s, e, w, h = KP[side]

    elbow = np.nan
    if min(conf[s], conf[e], conf[w]) >= MIN_CONF:
        elbow = angle_at(xy[s], xy[e], xy[w])
    tilt = np.nan
    if min(conf[s], conf[h]) >= MIN_CONF:
        tilt = tilt_from_horizontal(xy[s], xy[h])
    return elbow, tilt


def process_video(path, model):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Could not open {path.name}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{path.name}: {total} frames at {fps:.1f} fps")

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        el_l, tilt_l = measure(r, "left")
        el_r, tilt_r = measure(r, "right")
        tilt = tilt_r if MAIN_SIDE == "right" else tilt_l
        rows.append((idx, idx / fps, el_l, el_r, tilt))
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx}/{total}")
    cap.release()
    if not rows:
        print("No frames read.")
        return

    out = OUT_DIR / f"{path.stem}_{IMGSZ}"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "angles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "elbow_left", "elbow_right", "torso_tilt"])
        for fr, t, l, rr, tl in rows:
            w.writerow([fr, round(t, 4)] + ["" if np.isnan(v) else round(v, 2)
                                            for v in (l, rr, tl)])

    t = np.array([r[1] for r in rows])
    left = np.array([r[2] for r in rows])
    right = np.array([r[3] for r in rows])
    tilt = np.array([r[4] for r in rows])
    main = right if MAIN_SIDE == "right" else left

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t, left, label="left elbow",
             lw=2.5 if MAIN_SIDE == "left" else 1, alpha=1 if MAIN_SIDE == "left" else 0.5)
    ax1.plot(t, right, label="right elbow",
             lw=2.5 if MAIN_SIDE == "right" else 1, alpha=1 if MAIN_SIDE == "right" else 0.5)
    ax1.set_ylabel("elbow angle (deg)")
    ax1.set_title(f"{path.name}  imgsz={IMGSZ}  min_conf={MIN_CONF}")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax2.plot(t, tilt, color="gray")
    ax2.set_ylabel("torso tilt (deg)")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=120)
    plt.close(fig)

    valid = main[~np.isnan(main)]
    print(f"  usable frames ({MAIN_SIDE}): {100 * len(valid) / len(rows):.0f}%")
    for name, arr in (("left", left), ("right", right)):
        v = arr[~np.isnan(arr)]
        if len(v):
            print(f"  {name:>5}: usable {100 * len(v) / len(arr):3.0f}%  "
                  f"p5/p50/p95 = {np.percentile(v, 5):.0f}/"
                  f"{np.percentile(v, 50):.0f}/{np.percentile(v, 95):.0f}")
    print(f"  saved to {out}/")


def main():
    if len(sys.argv) > 1:
        videos = [Path(a) for a in sys.argv[1:]]
    else:
        videos = sorted(p for p in VIDEO_DIR.iterdir()
                        if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return
    model = YOLO(MODEL_PATH)
    for v in videos:
        process_video(v, model)


if __name__ == "__main__":
    main()