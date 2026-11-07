"""Render an annotated pushup demo video.

usage:
    python demo_video.py clip.mp4 [more.mp4 ...] [--weight 70] [--imgsz 416] [--max-side 1280]

Pass 1 runs the YOLO pose model over the whole clip and caches the keypoints.
Pass 2 re-reads the clip, runs the rep counter frame by frame, and draws the
skeleton, the elbow angle, a live angle graph and a stats bar. A summary card
holds for two seconds at the end. With ffmpeg on your PATH the frames are
encoded straight to H.264 (plays smoothly everywhere, LinkedIn included).
"""
import argparse
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import counter  # your rep counter (counter.py)

# ---- measurement (same rules as the logging script) ----
MIN_CONF = 0.5        # keypoint confidence needed for the angle maths
DRAW_CONF = 0.3       # keypoint confidence needed to draw a joint
MET = 8.0             # rough MET for vigorous push-ups; the kcal figure is only an estimate
GRAPH_WINDOW_S = 4.0  # seconds of history in the live graph

# COCO keypoints: 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
ARM = {"left": (5, 7, 9), "right": (6, 8, 10)}
HIP = {"left": 11, "right": 12}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]

# ---- palette (BGR) ----
WHITE = (245, 245, 245)
GRAY = (165, 165, 170)
INK = (14, 14, 16)
ACCENT = (255, 190, 40)
GREEN = (120, 230, 110)
ORANGE = (50, 150, 255)


# ------------------------------------------------------------------ maths
def angle_at(a, b, c):
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))))


def tilt_from_horizontal(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    deg = abs(np.degrees(np.arctan2(dy, dx)))
    return float(min(deg, 180 - deg))


def measure(xy, conf):
    """{side: (elbow_angle, torso_tilt)} with NaN where a keypoint is unreliable."""
    out = {}
    for side, (s, e, w) in ARM.items():
        h = HIP[side]
        elbow = tilt = np.nan
        if min(conf[s], conf[e], conf[w]) >= MIN_CONF:
            elbow = angle_at(xy[s], xy[e], xy[w])
        if min(conf[s], conf[h]) >= MIN_CONF:
            tilt = tilt_from_horizontal(xy[s], xy[h])
        out[side] = (elbow, tilt)
    return out


# ------------------------------------------------------------------ pass 1
def analyze(path, model, imgsz, device):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kps, left, right, tilt_l, tilt_r, stamps = [], [], [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not kps:
            print(f"  source: {frame.shape[1]}x{frame.shape[0]} @ {fps:.1f} fps, {total} frames")
        stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        r = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
        kp = None
        if r.keypoints is not None and r.keypoints.conf is not None and r.keypoints.xy.shape[0]:
            b = r.boxes.xyxy
            i = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())  # largest person
            kp = (r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy())
        m = measure(*kp) if kp else {"left": (np.nan, np.nan), "right": (np.nan, np.nan)}
        kps.append(kp)
        left.append(m["left"][0])
        right.append(m["right"][0])
        tilt_l.append(m["left"][1])
        tilt_r.append(m["right"][1])
        if len(kps) % 100 == 0:
            print(f"  pose {len(kps)}/{total}")
    cap.release()
    if len(stamps) > 10 and stamps[-1] > stamps[0]:
        eff = (len(stamps) - 1) / ((stamps[-1] - stamps[0]) / 1000.0)
        if abs(eff / fps - 1) > 0.05:
            print(f"  warning: header says {fps:.1f} fps but timestamps average {eff:.1f} fps "
                  "(variable frame rate). Convert the source to a constant rate first.")
    return {"fps": fps, "kps": kps, "left": np.array(left), "right": np.array(right),
            "tilt_left": np.array(tilt_l), "tilt_right": np.array(tilt_r)}


# ------------------------------------------------------------------ drawing helpers
def put(img, text, x, y, px, color=WHITE, thick=None, anchor="l", shadow=True):
    """Anti-aliased text with `px` pixel cap height; y is the baseline.
    A trailing degree sign is drawn as a small ring (Hershey fonts have none)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    deg = text.endswith("°")
    if deg:
        text = text[:-1]
    scale = max(px, 6) / 21.0
    th = thick or max(1, int(round(px / 9)))
    (w, _), _ = cv2.getTextSize(text, font, scale, th)
    total = w + (int(px * 0.5) if deg else 0)
    x, y = int(x), int(y)
    if anchor == "c":
        x -= total // 2
    elif anchor == "r":
        x -= total
    if shadow:
        cv2.putText(img, text, (x + 1, y + 2), font, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)
    if deg:
        cv2.circle(img, (x + w + int(px * 0.22), y - int(px * 0.82)),
                   max(2, int(px * 0.14)), color, max(1, th // 2), cv2.LINE_AA)
    return total


def panel(img, x1, y1, x2, y2, color=(20, 20, 22), alpha=0.6, r=16):
    """Translucent rounded rectangle."""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    w, h = x2 - x1, y2 - y1
    if w <= 2 or h <= 2:
        return
    r = max(1, min(int(r), w // 2, h // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    roi = img[y1:y2, x1:x2]
    layer = np.full_like(roi, color)
    m = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    roi[:] = (roi * (1 - m) + layer * m).astype(np.uint8)


def pill(img, text, cx, cy, w, h, color, px):
    panel(img, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, color=color, alpha=0.95, r=h / 2)
    put(img, text, cx, cy + px * 0.5, px, INK, anchor="c", shadow=False)


def draw_skeleton(img, kp, side, color, u):
    xy, conf = kp
    pts = {j: (int(xy[j][0]), int(xy[j][1])) for j in range(17) if conf[j] >= DRAW_CONF}
    tracked = {(5, 7), (7, 9)} if side == "left" else {(6, 8), (8, 10)}
    base = max(2, int(4 * u))
    for a, b in SKELETON:                      # dark outline first so it reads on any footage
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], INK, base + max(2, int(4 * u)), cv2.LINE_AA)
    for a, b in SKELETON:
        if a in pts and b in pts:
            hot = (a, b) in tracked
            cv2.line(img, pts[a], pts[b], color if hot else (235, 235, 235),
                     base + (int(2 * u) if hot else 0), cv2.LINE_AA)
    for j, p in pts.items():
        if j in (1, 2, 3, 4):                  # skip eyes and ears
            continue
        hot = j in ARM[side]
        cv2.circle(img, p, int((7.5 if hot else 5.5) * u), INK, -1, cv2.LINE_AA)
        cv2.circle(img, p, int((5 if hot else 3.5) * u), color if hot else WHITE, -1, cv2.LINE_AA)
    if 0 in pts:                               # head marker
        cv2.circle(img, pts[0], int(16 * u), INK, max(2, int(4 * u)), cv2.LINE_AA)
        cv2.circle(img, pts[0], int(16 * u), (235, 235, 235), max(1, int(2 * u)), cv2.LINE_AA)


def draw_elbow_arc(img, kp, side, color, label, u):
    xy, conf = kp
    s, e, w = ARM[side]
    if min(conf[s], conf[e], conf[w]) < DRAW_CONF:
        return None
    c = (int(xy[e][0]), int(xy[e][1]))
    v1, v2 = xy[s] - xy[e], xy[w] - xy[e]
    a1 = float(np.degrees(np.arctan2(v1[1], v1[0])) % 360)
    a2 = float(np.degrees(np.arctan2(v2[1], v2[0])) % 360)
    diff = (a2 - a1) % 360
    if diff > 180:                             # draw the short way round
        a1, diff = a2, 360 - diff
    r = int(40 * u)
    cv2.ellipse(img, c, (r, r), 0, a1, a1 + diff, INK, max(3, int(6 * u)), cv2.LINE_AA)
    cv2.ellipse(img, c, (r, r), 0, a1, a1 + diff, color, max(2, int(3 * u)), cv2.LINE_AA)
    if label:
        mid = np.radians(a1 + diff / 2)
        tx = c[0] + np.cos(mid) * (r + 32 * u)
        ty = c[1] + np.sin(mid) * (r + 32 * u)
        put(img, label, tx, ty + 9 * u, 22 * u, color, anchor="c")
    return c


def draw_graph(img, hist, rep_times, t, box, u):
    x1, y1, x2, y2 = box
    panel(img, x1, y1, x2, y2, alpha=0.6, r=14 * u)
    pad = int(12 * u)
    put(img, "ELBOW ANGLE", x1 + pad, y1 + 22 * u, 11 * u, GRAY, anchor="l", shadow=False)
    gx1, gx2 = x1 + pad, x2 - pad - int(24 * u)       # right gutter holds the threshold numbers
    gy1, gy2 = y1 + int(34 * u), y2 - pad
    lo, hi = 50.0, 190.0

    def ymap(a):
        return int(gy2 - (np.clip(a, lo, hi) - lo) / (hi - lo) * (gy2 - gy1))

    def xmap(tt):
        return int(gx1 + (tt - (t - GRAPH_WINDOW_S)) / GRAPH_WINDOW_S * (gx2 - gx1))

    for thr, col, dy in ((counter.UP_ANGLE, GREEN, -3), (counter.DOWN_ANGLE, ORANGE, 12)):
        y = ymap(thr)
        for xs in range(gx1, gx2, max(4, int(10 * u))):        # dashed threshold line
            cv2.line(img, (xs, y), (min(xs + max(2, int(5 * u)), gx2), y), col, 1, cv2.LINE_AA)
        put(img, str(int(thr)), x2 - pad, y + dy * u, 9 * u, col, anchor="r", shadow=False)
    pts = list(hist)
    for (t0, a0), (t1, a1) in zip(pts, pts[1:]):
        if np.isfinite(a0) and np.isfinite(a1) and t1 - t0 < 0.2:
            cv2.line(img, (xmap(t0), ymap(a0)), (xmap(t1), ymap(a1)), ACCENT,
                     max(2, int(2.5 * u)), cv2.LINE_AA)
    for rt in rep_times:                                        # a dot per counted rep
        if t - GRAPH_WINDOW_S <= rt <= t:
            cv2.circle(img, (xmap(rt), ymap(counter.UP_ANGLE)), int(4.5 * u), GREEN, -1, cv2.LINE_AA)
    if pts and np.isfinite(pts[-1][1]) and t - pts[-1][0] < 0.2:
        cv2.circle(img, (xmap(pts[-1][0]), ymap(pts[-1][1])), int(5 * u), WHITE, -1, cv2.LINE_AA)


# ------------------------------------------------------------------ stats
def kcal_estimate(rc, weight):
    """Very rough: MET x body weight x active time. Treat it as an estimate."""
    if rc.count == 0:
        return 0.0
    rts = rc.rep_times
    per = float(np.mean(np.diff(rts))) if len(rts) > 1 else 1.0
    active = (rts[-1] - rts[0]) + per
    return MET * weight * active / 3600.0


def tempo_last(rc):
    return rc.rep_times[-1] - rc.rep_times[-2] if len(rc.rep_times) > 1 else None


def draw_bar(img, rc, sm, state_col, pop, weight, u):
    H, W = img.shape[:2]
    compact = W / u < 760                       # narrow (portrait) frames get a two-panel layout
    m, bh = int(16 * u), int(112 * u)
    x1, y1, x2, y2 = m, H - m - bh, W - m, H - m
    panel(img, x1, y1, x2, y2, alpha=0.66, r=20 * u)

    tempo = tempo_last(rc)
    stats = (("TEMPO", f"{tempo:.1f} s/rep" if tempo else "--"),
             ("DEPTH", f"{rc.rep_depths[-1]:.0f}°" if rc.rep_depths else "--"),
             ("ENERGY", f"~{kcal_estimate(rc, weight):.1f} kcal"))

    # left: state pill + live elbow angle
    lx = x1 + int(22 * u)
    pill(img, rc.state, lx + 62 * u, y1 + 30 * u, 124 * u, 34 * u, state_col, 17 * u)
    put(img, "ELBOW", lx, y1 + 76 * u, 11 * u, GRAY, shadow=False)
    put(img, f"{sm:.0f}°" if sm is not None else "--", lx + 66 * u, y1 + 100 * u, 28 * u)

    if compact:
        cx = (lx + 150 * u + x2 - 22 * u) / 2   # centre the count in the space right of the pill
        sy2 = y1 - int(8 * u)                   # stats move to their own strip above the bar
        sy1 = sy2 - int(58 * u)
        panel(img, x1, sy1, x2, sy2, alpha=0.66, r=16 * u)
        for k, (name, val) in enumerate(stats):
            c = x1 + (x2 - x1) * (1 + 2 * k) / 6
            put(img, name, c, sy1 + 22 * u, 11 * u, GRAY, anchor="c", shadow=False)
            put(img, val, c, sy1 + 47 * u, 18 * u, WHITE, anchor="c")
    else:
        cx = (x1 + x2) // 2
        rx = x2 - int(22 * u)
        lab = rx - int(210 * u)
        for k, (name, val) in enumerate(stats):
            y = y1 + int((34 + 29 * k) * u)
            put(img, name, lab, y, 11 * u, GRAY, shadow=False)
            put(img, val, rx, y, 19 * u, WHITE, anchor="r")

    # centre: the big rep count
    put(img, "REPS", cx, y1 + 27 * u, 14 * u, GRAY, anchor="c", shadow=False)
    col = tuple(int(a + (b - a) * pop) for a, b in zip(WHITE, GREEN))   # flashes green after a rep
    put(img, str(rc.count), cx, y1 + 103 * u, 66 * u * (1 + 0.18 * pop), col, anchor="c")


def draw_title(img, side, model_name, u):
    m = int(16 * u)
    w = int(min(img.shape[1] * 0.47, 278 * u))
    panel(img, m, m, m + w, m + 62 * u, alpha=0.6, r=14 * u)
    put(img, "PUSHUP COUNTER", m + 16 * u, m + 27 * u, 17 * u, WHITE)
    put(img, f"{model_name} | {side} arm", m + 16 * u, m + 49 * u, 11 * u, GRAY, shadow=False)


def summary_frame(base, rc, weight, u):
    out = base.copy()
    H, W = out.shape[:2]
    cw, ch = int(min(W * 0.84, 540 * u)), int(300 * u)
    x1, y1 = (W - cw) // 2, (H - ch) // 2
    panel(out, x1, y1, x1 + cw, y1 + ch, color=(18, 18, 20), alpha=0.88, r=24 * u)
    put(out, "SET COMPLETE", W // 2, y1 + 42 * u, 16 * u, GRAY, anchor="c", shadow=False)
    put(out, str(rc.count), W // 2, y1 + 146 * u, 92 * u, WHITE, anchor="c")
    put(out, "REPS", W // 2, y1 + 176 * u, 16 * u, ACCENT, anchor="c", shadow=False)
    depth = f"{np.mean(rc.rep_depths):.0f}°" if rc.rep_depths else "--"
    tempo = f"{np.mean(np.diff(rc.rep_times)):.1f} s" if len(rc.rep_times) > 1 else "--"
    stats = (("AVG DEPTH", depth), ("AVG TEMPO", tempo),
             ("ENERGY", f"~{kcal_estimate(rc, weight):.1f} kcal"))
    for k, (name, val) in enumerate(stats):
        cx = x1 + int(cw * (1 + 2 * k) / 6)
        put(out, name, cx, y1 + 230 * u, 11 * u, GRAY, anchor="c", shadow=False)
        put(out, val, cx, y1 + 262 * u, 22 * u, WHITE, anchor="c")
    return out


# ------------------------------------------------------------------ pass 2
def render(video, out_path, data, side, weight, model_name, max_side=1280):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No frames to render.")
    H0, W0 = frame.shape[:2]
    scale = min(1.0, max_side / max(W0, H0))             # big phone/4K clips get downscaled
    W, H = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2   # H.264 needs even sizes
    sx, sy = W / W0, H / H0
    fps = data["fps"]
    writer = VideoSink(out_path, W, H, fps)
    print(f"  output: {W}x{H} @ {fps:.1f} fps")
    u = min(W, H) / 540.0

    angles = data[side]
    tilts = data["tilt_" + side]
    rc = counter.RepCounter(max(1, round(counter.SMOOTH_S * fps)))
    hist = deque()
    last_rep_i = None
    last_frame = None
    i = 0
    while ok:
        t = i / fps
        prev = rc.count
        sm = rc.update(t, angles[i], tilts[i]) if i < len(angles) else None
        if rc.count > prev:
            last_rep_i = i
        hist.append((t, np.nan if sm is None else sm))
        while hist and hist[0][0] < t - GRAPH_WINDOW_S - 0.1:
            hist.popleft()

        since = (i - last_rep_i) / fps if last_rep_i is not None else 99.0
        pop = max(0.0, 1.0 - since / 0.35)                # 1 -> 0 over 0.35 s after each rep
        state_col = GREEN if rc.state == "UP" else ORANGE

        if scale < 1.0:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        img = (frame.astype(np.float32) * 0.8).astype(np.uint8)   # dim footage so overlays pop
        kp = data["kps"][i] if i < len(data["kps"]) else None
        if kp is not None and scale < 1.0:
            kp = (kp[0] * np.array([sx, sy]), kp[1])
        if kp is not None:
            draw_skeleton(img, kp, side, state_col, u)
            label = f"{sm:.0f}°" if sm is not None else None
            elbow = draw_elbow_arc(img, kp, side, state_col, label, u)
            if elbow is not None and since < 0.5:                  # ring burst on each rep
                p = since / 0.5
                layer = img.copy()
                cv2.circle(layer, elbow, int((34 + 56 * p) * u), GREEN, max(2, int(5 * u * (1 - p))),
                           cv2.LINE_AA)
                cv2.addWeighted(layer, 1 - p, img, p, 0, img)

        draw_title(img, side, model_name, u)
        gw = int(min(W * 0.42, 290 * u))
        gm = int(16 * u)
        draw_graph(img, hist, rc.rep_times, t, (W - gm - gw, gm, W - gm, gm + int(118 * u)), u)
        draw_bar(img, rc, sm, state_col, pop, weight, u)

        writer.write(img)
        last_frame = img
        ok, frame = cap.read()
        i += 1
    cap.release()

    if last_frame is not None:                              # outro: summary card, fades in
        dimmed = (last_frame.astype(np.float32) * 0.45).astype(np.uint8)
        card = summary_frame(dimmed, rc, weight, u)
        n = int(2.0 * fps)
        for k in range(n):
            f = min(1.0, k / (0.5 * fps))
            writer.write(cv2.addWeighted(card, f, last_frame, 1 - f, 0))
    writer.close()
    return rc


class VideoSink:
    """Pipes frames straight into an ffmpeg H.264 encoder when ffmpeg is installed
    (small file, plays smoothly everywhere); otherwise falls back to OpenCV's mp4v
    writer, which is bulky and plays poorly in many players."""

    def __init__(self, path, w, h, fps):
        self.proc = self.writer = None
        ff = shutil.which("ffmpeg")
        if ff:
            cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{fps:.5f}", "-i", "-", "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print("  ffmpeg not found: using the mp4v fallback, which is bulky and often laggy.\n"
                  "  Install ffmpeg (e.g. winget install Gyan.FFmpeg) for smooth H.264 output.")

    def write(self, img):
        if self.proc:
            try:
                self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
            except BrokenPipeError:
                raise SystemExit("ffmpeg stopped unexpectedly; run it by hand to see why.")
        else:
            self.writer.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.writer.release()


def main():
    ap = argparse.ArgumentParser(description="Render an annotated pushup demo video.")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--weight", type=float, default=70.0,
                    help="body weight in kg, used only for the calorie estimate")
    ap.add_argument("--out-dir", default="demos")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="downscale so the longer side is at most this many pixels")
    args = ap.parse_args()

    from ultralytics import YOLO      # imported here so the drawing code stays testable without it
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model_name = Path(args.model).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for v in args.videos:
        v = Path(v)
        print(f"\n{v.name}: running pose model...")
        data = analyze(v, model, args.imgsz, device)
        if not data["kps"]:
            print("  no frames read, skipping")
            continue
        t = np.arange(len(data["left"])) / data["fps"]
        side = counter.choose_side(t, data["left"], data["right"])
        print(f"  tracking the {side} arm; rendering...")
        final = out_dir / f"{v.stem}_demo.mp4"
        rc = render(v, final, data, side, args.weight, model_name, args.max_side)
        print(f"  {rc.count} reps -> {final} ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()