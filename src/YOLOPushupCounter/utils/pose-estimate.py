import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# ---- config: change these between runs ----
MODEL_PATH = "yolo26n-pose.pt"   # or "yolo26n-pose_openvino_model/"
IMGSZ = 320
SKIP = 1             # process every Nth frame
DEVICE = 0           # use "cpu" for OpenVINO exports
WARMUP = 10          # processed frames to ignore in the stats
MAX_PROCESSED = 200  # only used when display is off

model = YOLO(MODEL_PATH)

# ---- pick source ----
choice = input("0 for webcam, 1 for list of videos in train folder\n")
if choice == "0":
    cap = cv2.VideoCapture(0)
else:
    videos = sorted(p for p in (Path.cwd() / "train-videos").iterdir() if p.is_file())
    for i, v in enumerate(videos, start=1):
        print(i, v.name)
    cap = cv2.VideoCapture(str(videos[int(input("Which video do you want?\n")) - 1]))

show = input("Display on? (y/n)\n").strip().lower() == "y"

if show:
    cv2.namedWindow("YOLO Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Pose", 960, 540)

# ---- measurement state ----
gaps = deque(maxlen=30)  # seconds between processed frames
stats = {k: [] for k in ("read", "pre", "infer", "post", "plot", "show")}
frame_idx = 0
processed = 0
last_done = None
t_measure_start = None
read_start = time.perf_counter()

while cap.isOpened():
    # grab() advances without decoding, so skipped frames are cheap
    if not cap.grab():
        break
    frame_idx += 1
    if frame_idx % SKIP != 0:
        continue
    ok, frame = cap.retrieve()  # decode only the frames we keep
    if not ok:
        break
    read_ms = (time.perf_counter() - read_start) * 1000  # includes skipped grabs

    # rolling FPS from the gap between processed frames
    now = time.perf_counter()
    if last_done is not None:
        gaps.append(now - last_done)
    last_done = now
    fps = len(gaps) / sum(gaps) if gaps else 0.0

    results = model.predict(frame, imgsz=IMGSZ, device=DEVICE, half=True, verbose=False)
    sp = results[0].speed  # ms: preprocess, inference, postprocess

    plot_ms = show_ms = 0.0
    key = -1
    if show:
        t = time.perf_counter()
        annotated = results[0].plot()
        cv2.putText(annotated, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        plot_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        cv2.imshow("YOLO Pose", annotated)
        key = cv2.waitKey(1) & 0xFF
        show_ms = (time.perf_counter() - t) * 1000

    processed += 1
    if processed == WARMUP:
        t_measure_start = time.perf_counter()
    if processed > WARMUP:
        stats["read"].append(read_ms)
        stats["pre"].append(sp["preprocess"])
        stats["infer"].append(sp["inference"])
        stats["post"].append(sp["postprocess"])
        stats["plot"].append(plot_ms)
        stats["show"].append(show_ms)

    if show and key == ord("q"):
        break
    if not show and processed >= MAX_PROCESSED:
        break

    read_start = time.perf_counter()  # restart the read timer

cap.release()
cv2.destroyAllWindows()

# ---- summary ----
n = len(stats["infer"])
if n:
    elapsed = time.perf_counter() - t_measure_start
    print(f"\nconfig: imgsz={IMGSZ} skip={SKIP} model={MODEL_PATH} display={show}")
    print(f"measured frames: {n}")
    print(f"end-to-end processing FPS: {n / elapsed:.1f}")
    for k, v in stats.items():
        print(f"{k:>6}: {sum(v) / len(v):7.2f} ms")
    model_ms = (sum(stats["pre"]) + sum(stats["infer"]) + sum(stats["post"])) / n
    print(f"model-only ceiling: {1000 / model_ms:.1f} FPS")
else:
    print("Not enough frames to measure (need more than WARMUP).")