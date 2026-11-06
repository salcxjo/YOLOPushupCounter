import csv
import statistics
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file, no window
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---- config: change between runs ----
MODEL_PATH = "yolo26n-pose.pt"
IMGSZ = 640
DEVICE = 0
SIDE = "right"      # whichever arm faces the camera
MIN_CONF = 0.5     # ignore frames where any of the 3 keypoints is below this
TRIM_S = 0.5       # seconds to drop at each end of a recording

# COCO keypoint indices: (shoulder, elbow, wrist)
KP = {"left": (5, 7, 9), "right": (6, 8, 10)}


def angle_at(a, b, c):
    """Angle in degrees at point b, formed by a-b-c."""
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    cos = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def elbow_angle(result, side, min_conf):
    """Elbow angle for the largest person in frame, or None if unusable."""
    kps = result.keypoints
    if kps is None or kps.conf is None or kps.xy.shape[0] == 0:
        return None
    boxes = result.boxes.xyxy
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(areas.argmax())
    xy = kps.xy[i].cpu().numpy()
    conf = kps.conf[i].cpu().numpy()
    s, e, w = KP[side]
    if min(conf[s], conf[e], conf[w]) < min_conf:
        return None
    return angle_at(xy[s], xy[e], xy[w])


def summarize(samples, total_frames):
    """samples: list of (t, angle). Prints stats and saves CSV + plot."""
    if len(samples) < 2:
        print("Not enough valid frames.")
        return
    t0, t1 = samples[0][0], samples[-1][0]
    kept = [(t, a) for t, a in samples if t0 + TRIM_S <= t <= t1 - TRIM_S]
    if len(kept) < 2:
        print("Recording too short after trimming.")
        return
    angles = [a for _, a in kept]
    print(f"\nimgsz={IMGSZ} side={SIDE} min_conf={MIN_CONF}")
    print(f"frames in recording: {total_frames}, valid: {len(samples)} "
          f"({100 * len(samples) / total_frames:.0f}% usable)")
    print(f"after trim: n={len(angles)}")
    print(f"mean  : {statistics.mean(angles):6.2f} deg")
    print(f"stdev : {statistics.pstdev(angles):6.2f} deg   <-- stillness spread")
    print(f"min   : {min(angles):6.2f}  max: {max(angles):6.2f}  "
          f"range: {max(angles) - min(angles):6.2f}")

    stamp = time.strftime("%H%M%S")
    base = f"hold_{IMGSZ}_{SIDE}_{stamp}"
    with open(base + ".csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "angle_deg"])
        wr.writerows((round(t - t0, 3), round(a, 2)) for t, a in samples)

    plt.figure(figsize=(8, 3))
    plt.plot([t - t0 for t, _ in samples], [a for _, a in samples])
    plt.xlabel("time (s)")
    plt.ylabel("elbow angle (deg)")
    plt.title(f"imgsz={IMGSZ}  stdev={statistics.pstdev(angles):.2f}")
    plt.tight_layout()
    plt.savefig(base + ".png")
    plt.close()
    print(f"saved {base}.csv and {base}.png")


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hold test", cv2.WINDOW_NORMAL)

recording = False
samples, total_frames = [], 0
print("SPACE = start/stop recording, q = quit")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    result = model.predict(frame, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    ang = elbow_angle(result, SIDE, MIN_CONF)

    if recording:
        total_frames += 1
        if ang is not None:
            samples.append((time.perf_counter(), ang))

    view = result.plot()
    label = f"{ang:.0f} deg" if ang is not None else "no reading"
    cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if recording:
        cv2.putText(view, "REC", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Hold test", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        if not recording:
            samples, total_frames = [], 0
            recording = True
        else:
            recording = False
            summarize(samples, total_frames)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()