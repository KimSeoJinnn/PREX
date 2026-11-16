"""Automatic acquisition: detector + shape evidence + temporal stability."""

import math

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_landmarks import detect_landmarks


class AutomaticBottleFinder:
    def __init__(self, detector=None, interval=.30, stable_samples=3):
        self.detector = detector
        self.interval = interval
        self.stable_samples = stable_samples
        self.last_search = -math.inf
        self.last_model = -math.inf
        self.detections = []
        self.candidate = None
        self.count = 0
        self.status = "Searching for cap, body and base"
        self.model_error = None

    def reset(self):
        self.candidate = None
        self.count = 0
        self.last_search = -math.inf

    def update(self, frame, hand, now, expected=None):
        if now - self.last_search < self.interval:
            return None
        self.last_search = now
        if self.detector is not None and now - self.last_model >= .9:
            self.last_model = now
            try:
                self.detections = self.detector.detect(frame)
            except (cv2.error, ValueError, RuntimeError, OSError) as error:
                # The geometric detector continues if an optional DNN fails.
                self.model_error = str(error)
                self.detector = None
                self.detections = []
        if len(self.detections) > 1 and expected is None:
            self.count = 0
            self.candidate = None
            self.status = "Multiple bottles - show one bottle"
            return None
        proposals = []
        if len(self.detections) == 1:
            x, y, width, height = self.detections[0].region
            margin = max(8, round(width * .12))
            region = (x - margin, y - margin, width + margin * 2, height + margin * 2)
            proposals = detect_landmarks(frame, hand, region, expected)
        if not proposals:
            region = None
            if expected is not None:
                cap, base = expected
                length = float(np.linalg.norm(base - cap))
                minimum, maximum = np.minimum(cap, base), np.maximum(cap, base)
                margin = length * .35
                region = (minimum[0] - margin, minimum[1] - length * .12,
                          maximum[0] - minimum[0] + margin * 2,
                          maximum[1] - minimum[1] + length * .27)
            proposals = detect_landmarks(frame, hand, region, expected)
        if len(proposals) != 1:
            self.count = 0
            self.candidate = None
            self.status = ("Multiple bottle shapes - show one" if proposals else
                           "Searching - show visible cap and base")
            return None
        candidate = proposals[0]
        length = float(np.linalg.norm(candidate.base - candidate.cap))
        if self.candidate is not None:
            movement = max(np.linalg.norm(candidate.cap - self.candidate.cap),
                           np.linalg.norm(candidate.base - self.candidate.base))
            self.count = self.count + 1 if movement <= max(6, length * .035) else 1
        else:
            self.count = 1
        self.candidate = candidate
        self.status = f"Bottle found - hold still {min(self.count, self.stable_samples)}/{self.stable_samples}"
        if self.count < self.stable_samples:
            return None
        return candidate

    def acquire(self, frame, hand, now):
        candidate = self.update(frame, hand, now)
        if candidate is None:
            return None
        try:
            tracker = AxisBottleTracker(frame, candidate.cap, candidate.base, hand,
                                        candidate.body_width, candidate.source)
        except ValueError:
            self.status = "Keep the whole bottle inside the frame"
            self.reset()
            return None
        if tracker.last.width_px is None or tracker.last.quality < .35:
            self.status = "Cap/base found; body contour weak/hidden"
            return None
        self.status = "Automatic axis ready"
        self.reset()
        return tracker
