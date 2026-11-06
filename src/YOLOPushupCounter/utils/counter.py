from collections import deque

import numpy as np

# ---- tunable parameters (times in seconds, so they survive FPS changes) ----
DOWN_ANGLE = 110      # smoothed angle must go below this to count as "down"
UP_ANGLE = 150        # ...and back above this to complete the rep
SMOOTH_S = 0.15       # causal median filter length
DWELL_S = 0.10        # must stay below DOWN_ANGLE this long to enter DOWN
MIN_REP_GAP_S = 0.5   # ignore a rep completing sooner than this after the last
MAX_GAP_S = 0.4       # unreadable stretch longer than this discards the partial rep
TILT_MAX = 28         # torso more tilted than this = not in a pushup position
SIDE_WINDOW_S = 3.0   # choose the arm from the first few seconds only


class RepCounter:
    """Frame-by-frame rep counter. Only ever looks at the past, so the same
    logic works offline on a CSV and live on a webcam."""

    def __init__(self, smooth_frames):
        self.buf = deque(maxlen=max(1, smooth_frames))
        self.state = "UP"            # always start at the top
        self.below_since = None      # when the angle first dropped under DOWN_ANGLE
        self.last_valid_t = None
        self.last_rep_t = None
        self.count = 0
        self.rep_times = []

    def _reset_partial(self):
        """Forget any rep in progress. The running total is kept."""
        self.state = "UP"
        self.below_since = None
        self.buf.clear()

    def update(self, t, angle, tilt):
        if np.isnan(angle):
            return                                   # unreadable frame: wait it out

        if self.last_valid_t is not None and t - self.last_valid_t > MAX_GAP_S:
            self._reset_partial()                    # lost track too long to trust
        self.last_valid_t = t

        # NaN tilt means "no information", so it does not block counting
        if not np.isnan(tilt) and tilt > TILT_MAX:
            self._reset_partial()                    # not in a pushup position
            return

        self.buf.append(angle)
        a = float(np.median(self.buf))               # smoothed angle

        if self.state == "UP":
            if a < DOWN_ANGLE:
                if self.below_since is None:
                    self.below_since = t
                elif t - self.below_since >= DWELL_S:
                    self.state = "DOWN"              # held low long enough
            else:
                self.below_since = None
        else:  # DOWN
            if a > UP_ANGLE:
                self.state = "UP"
                self.below_since = None
                if self.last_rep_t is None or t - self.last_rep_t >= MIN_REP_GAP_S:
                    self.count += 1
                    self.rep_times.append(float(t))
                    self.last_rep_t = t


def choose_side(t, left, right):
    """The camera-facing arm has more readable frames. Uses only the first
    SIDE_WINDOW_S seconds, so the same rule works live."""
    in_window = (t - t[0]) <= SIDE_WINDOW_S
    n_left = int(np.sum(~np.isnan(left[in_window])))
    n_right = int(np.sum(~np.isnan(right[in_window])))
    return "left" if n_left > n_right else "right"


def count_reps(t, elbow_left, elbow_right, tilt):
    t = np.asarray(t, dtype=float)
    left = np.asarray(elbow_left, dtype=float)
    right = np.asarray(elbow_right, dtype=float)
    tilt = np.asarray(tilt, dtype=float)

    if len(t) < 2:
        return {"count": 0, "side": "right", "rep_times": []}

    side = choose_side(t, left, right)
    angles = left if side == "left" else right

    dt = float(np.median(np.diff(t)))
    smooth_frames = max(1, int(round(SMOOTH_S / dt))) if dt > 0 else 1

    counter = RepCounter(smooth_frames)
    for ti, ai, tilti in zip(t, angles, tilt):
        counter.update(ti, ai, tilti)

    return {"count": counter.count, "side": side, "rep_times": counter.rep_times}