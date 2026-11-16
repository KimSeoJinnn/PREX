"""Short image-supported hand tracking for the pink proxy, not joint angles.

Landmark outliers are checked against bidirectional optical flow. Flow-only
frames can keep a visual guide for at most 450 ms after a supported detection;
they must never supply a calibration sample, joint angle, or width reading.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GripTrack:
    points: list
    side: str | None
    source: str
    age_ms: float = 0.

    @property
    def display_only(self):
        return self.source == 'optical_flow'


class GripHandTracker:
    max_gap_s = .45

    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = self.points = self.stamp = self.detected_at = None
        self.side = None
        self.last = GripTrack([], None, 'missing')

    @staticmethod
    def _valid(points, shape):
        if points.shape != (21, 2) or not np.isfinite(points).all():
            return False
        h, w = shape
        return (len(np.unique(points, axis=0)) >= 8
                and np.linalg.norm(points[5] - points[17]) >= 12
                and np.all((points >= 0) & (points < (w, h))))

    def _flow(self, gray):
        if self.gray is None or self.gray.shape != gray.shape or self.points is None:
            return None
        first = self.points.astype(np.float32).reshape(-1, 1, 2)
        forward, ok, errors = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, first, None, winSize=(21, 21), maxLevel=2)
        if forward is None or ok is None or errors is None or not np.isfinite(forward).all():
            return None
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, forward, None, winSize=(21, 21), maxLevel=2)
        if backward is None or back_ok is None:
            return None
        xy = forward.reshape(-1, 2)
        good = (ok.ravel() > 0) & (back_ok.ravel() > 0)
        good &= np.linalg.norm(backward - first, axis=2).ravel() < 1.2
        good &= errors.ravel() < 25
        good &= np.all((xy >= 0) & (xy < gray.shape[::-1]), axis=1)
        if np.count_nonzero(good) < 8:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            self.points[good], xy[good], method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if matrix is None or inliers is None or np.mean(inliers) < .6:
            return None
        support = self.points[good][inliers.ravel() > 0]
        if len(support) < 6 or np.min(np.ptp(support, axis=0)) < 15:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        rotation = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
        rigid = cv2.transform(first, matrix).reshape(-1, 2)
        if (not .85 <= scale <= 1.18 or rotation > 20
                or np.linalg.norm(np.median(rigid - self.points, axis=0)) > 45):
            return None
        # Locally supported fingers can move independently of the palm. Any
        # unsupported point uses the palm transform for the visual guide only.
        predicted = rigid.copy()
        predicted[good] = xy[good]
        return predicted if self._valid(predicted, gray.shape) else None

    def update(self, frame, hand, side, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = np.asarray(hand, np.float32)
        raw_ok = self._valid(raw, gray.shape)
        if now is None:
            return GripTrack(raw.tolist() if raw_ok else [], side, 'landmarks' if raw_ok else 'missing')
        if self.stamp is not None and now == self.stamp:
            return self.last
        if self.stamp is not None and (now < self.stamp or now - self.stamp > self.max_gap_s):
            self.reset()
        age = float('inf') if self.detected_at is None else now - self.detected_at
        predicted = self._flow(gray) if age <= self.max_gap_s else None
        accept_raw = raw_ok
        if raw_ok and predicted is not None:
            palm = max(20., float(np.linalg.norm(predicted[5] - predicted[17])))
            displacement = raw - predicted
            shift = np.linalg.norm(np.median(displacement[[0, 5, 9, 13, 17]], axis=0))
            jump = np.percentile(np.linalg.norm(displacement, axis=1), 80)
            accept_raw = shift <= max(9., palm * .20) and jump <= max(12., palm * .28)
        if accept_raw:
            # Correct slow landmark drift without allowing detector jitter to
            # replace a well-supported image track in one step.
            points = raw if predicted is None else predicted * .65 + raw * .35
            stable_side = self.side if predicted is not None and self.side is not None else side
            self.detected_at = now
            self.last = GripTrack(points.tolist(), stable_side, 'landmarks')
        elif predicted is not None:
            points = predicted
            self.last = GripTrack(points.tolist(), self.side, 'optical_flow', age * 1000)
        else:
            self.reset()
            return self.last
        self.gray, self.points, self.stamp, self.side = gray, points, now, self.last.side
        return self.last
