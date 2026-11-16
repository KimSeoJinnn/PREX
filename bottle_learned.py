"""Learned bottle localization; contour visibility is a separate outcome."""

import time

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker, EndpointTracker, BottlePose, propose_endpoints
from bottle_detector import Detection
from bottle_landmarks import Landmarks
from bottle_outline import hand_mask
from bottle_endpoints import cap_identity, same_cap, observe_endpoints, foot_patch, same_foot


def detection_landmarks(frame, detection, hand):
    x, y, width, height = detection.region
    mask = getattr(detection, "mask", None)
    if mask is not None and np.count_nonzero(mask) >= 40:
        yy, xx = np.nonzero(mask)
        points = np.column_stack((xx, yy)).astype(float)
        center = np.mean(points, axis=0)
        _, vectors = np.linalg.eigh(np.cov((points - center).T))
        direction = vectors[:, -1]
        if direction[1] < 0:
            direction = -direction
        normal = np.array([direction[1], -direction[0]])
        along, across = (points - center) @ direction, (points - center) @ normal
        # Use occupied longitudinal bins, not pixel-mass percentiles, to keep
        # a narrow cap from being swallowed by the much wider body.
        low, high = float(along.min()), float(along.max())
        span = high - low
        cap_part, base_part = along <= low + span * .10, along >= high - span * .08
        cap = center + direction * (low + span * .04) + normal * np.median(across[cap_part])
        base = center + direction * (high - span * .025) + normal * np.median(across[base_part])
        body = (along > low + span * .30) & (along < low + span * .85)
        body_width = float(np.percentile(across[body], 95) - np.percentile(across[body], 5)) if body.any() else width
        source = "learned_instance_mask"
    else:
        cap, base, _ = propose_endpoints(frame, detection, hand)
        # A generic box around a gripped bottle often includes protruding
        # fingers. Start slightly inside it; the edge search still determines
        # the measured width and can expand beyond this initialization hint.
        body_width, source = float(width) * .85, "learned_bottle_box"
    cap = np.clip(cap, (7, 7), (frame.shape[1] - 8, frame.shape[0] - 8))
    base = np.clip(base, (7, 7), (frame.shape[1] - 8, frame.shape[0] - 8))
    length = float(np.linalg.norm(base - cap))
    if length < 80:
        return None
    proposal = Landmarks(cap, base, max(25., body_width), float(detection.score), source)
    if source == 'learned_bottle_box':
        observed = observe_endpoints(frame, proposal, hand_mask(frame.shape, hand), region=detection.region)
        if observed is not None:
            return observed
    return proposal


class LearnedBottleFinder:
    def __init__(self, detector, interval=.8, stable_samples=2):
        self.detector = detector
        self.interval = interval
        self.stable_samples = stable_samples
        self.last_search = -float("inf")
        self.inference_seconds = 0.
        self.detection = None
        self.detection_time = -float("inf")
        self.candidate = None
        self.count = 0
        self.status = "Searching for a bottle with AI"
        self._crop_next = True
        self._focus_region = None
        self._candidate_frame = None
        self._candidate_hand = []

    def reset(self):
        self.candidate = None
        self.count = 0
        self._focus_region = None
        self._candidate_frame = None
        self._candidate_hand = []

    @staticmethod
    def _choose(detections, hand, frame_shape, expected):
        # A cropped neck and the full occluded bottle can survive ordinary IoU
        # NMS together. Remove nested fragments before deciding there are two
        # bottles. Tiny background bottles cannot support our 80px axis either.
        usable = [d for d in detections if d.region[3] >= 90 and d.region[2] >= 25]
        detections = []
        for item in usable:
            x, y, w, h = item.region
            contained = False
            for other in usable:
                if other is item:
                    continue
                ox, oy, ow, oh = other.region
                intersection = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
                if h < oh * .78 and intersection / (w * h) > .85:
                    contained = True
                    break
            if not contained:
                detections.append(item)
        if expected is not None:
            midpoint = (expected[0] + expected[1]) / 2
            radius = np.linalg.norm(expected[1] - expected[0]) * .65
            detections = [d for d in detections if np.linalg.norm(
                np.array([d.region[0] + d.region[2] / 2, d.region[1] + d.region[3] / 2]) - midpoint) < radius]
        if hand:
            points = np.asarray(hand, float)
            palm = max(25., float(np.linalg.norm(points[5] - points[17]))) if len(points) == 21 else 45.
            nearby = []
            for item in detections:
                x, y, w, h = item.region
                distances = np.linalg.norm(np.maximum((x, y) - points, 0)
                                           + np.minimum((x + w, y + h) - points, 0), axis=1)
                if float(np.min(distances)) <= max(18., palm * .55):
                    nearby.append(item)
            detections = nearby
        if len(detections) == 1:
            return detections[0]
        if not detections or not hand:
            return None
        center = np.mean(hand, axis=0)
        ranked = []
        for detection in detections:
            x, y, w, h = detection.region
            distance = np.linalg.norm(np.maximum((x, y) - center, 0) + np.minimum((x + w, y + h) - center, 0))
            ranked.append((float(distance), detection))
        ranked.sort(key=lambda item: item[0])
        if ranked[0][0] + 25 < ranked[1][0]:
            return ranked[0][1]
        return None

    def _detect(self, frame, hand):
        # Alternate full view and hand-neighborhood zoom when the object is
        # small/occluded. Never run both expensive scans in one analysis tick.
        self._crop_next = not self._crop_next
        bounds = self._focus_region
        if bounds is None and self.candidate is None and self._crop_next and len(hand) == 21:
            points = np.asarray(hand)
            lo, hi = points.min(axis=0), points.max(axis=0)
            extent = np.maximum(hi - lo, (80, 100))
            lo = np.maximum(0, lo - extent * (1.0, 1.7)).astype(int)
            hi = np.minimum(frame.shape[1::-1], hi + extent * (1.0, 1.2)).astype(int)
            bounds = (*lo, *hi)
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            lo, hi = np.array([x0, y0], int), np.array([x1, y1], int)
            patch = frame[lo[1]:hi[1], lo[0]:hi[0]]
            found = self.detector.detect(patch)
            shifted = []
            for item in found:
                x, y, w, h = item.region
                mask = getattr(item, "mask", None)
                if mask is not None:
                    from bottle_segmentation import InstanceDetection
                    full = np.zeros(frame.shape[:2], np.uint8)
                    full[lo[1]:hi[1], lo[0]:hi[0]] = mask
                    shifted.append(InstanceDetection((x + int(lo[0]), y + int(lo[1]), w, h), item.score, full))
                else:
                    shifted.append(Detection((x + int(lo[0]), y + int(lo[1]), w, h), item.score))
            return shifted
        return self.detector.detect(frame)

    def _same_moving_bottle(self, frame, hand, candidate):
        """Stability means the same observed object, not immobility on screen."""
        previous = self.candidate
        if previous is None or self._candidate_frame is None:
            return False
        old = BottlePose(previous.cap, previous.base)
        new = BottlePose(candidate.cap, candidate.base)
        if not .75 <= new.length / old.length <= 1.33:
            return False
        if max(np.linalg.norm(new.cap - old.cap), np.linalg.norm(new.base - old.base)) > old.length * .60:
            return False
        old_mask = hand_mask(frame.shape, self._candidate_hand)
        new_mask = hand_mask(frame.shape, hand)
        if (previous.endpoint_quality is not None and candidate.endpoint_quality is not None
                and min(*previous.endpoint_quality, *candidate.endpoint_quality) >= .60):
            # Both searches independently observed the cap AND terminating
            # bottom rim. A changed grip/background can defeat optical flow
            # without making them different bottles. Confirm their material
            # appearance directly before demanding the old flow bridge again.
            if (same_cap(cap_identity(self._candidate_frame, old.cap, previous.body_width, old_mask),
                         cap_identity(frame, new.cap, candidate.body_width, new_mask))
                    and same_foot(foot_patch(self._candidate_frame, old.cap, old.base, previous.body_width),
                                  foot_patch(frame, new.cap, new.base, candidate.body_width))):
                return True
        tracker = EndpointTracker(self._candidate_frame, old,
                                  old_mask, previous.body_width)
        tracker.allow_geometry = False
        moved, _, _ = tracker.track(frame, new_mask)
        return moved is not None and max(np.linalg.norm(moved.cap - new.cap),
                                         np.linalg.norm(moved.base - new.base)) < max(12., new.length * .10)

    def update(self, frame, hand, now, expected=None):
        if self.detector is None:
            self.status = "Bottle model unavailable - check terminal"
            return None
        if now - self.last_search < max(self.interval, self.inference_seconds * 3):
            return None
        self.last_search = now
        started = time.perf_counter()
        detections = self._detect(frame, hand)
        self.inference_seconds = time.perf_counter() - started
        detection = self._choose(detections, hand, frame.shape, expected)
        self.detection, self.detection_time = detection, now
        if detection is None:
            self.reset()
            self.status = "Show one bottle clearly" if detections else "AI searching - move bottle closer"
            return None
        candidate = detection_landmarks(frame, detection, hand)
        if candidate is None:
            self.status = "Bottle located - show its full height"
            return None
        if np.linalg.norm(candidate.base - candidate.cap) < max(120, frame.shape[0] * .34):
            # A confident neck-only detection is not a whole PET bottle. Zoom
            # with enough context below the neck rather than calibrating its
            # lower end as the bottle base. No second DNN pass on this tick.
            x, y, w, h = detection.region
            cx = x + w / 2
            self._focus_region = (max(0, round(cx - w * 1.6)), max(0, round(y - h * .22)),
                                  min(frame.shape[1], round(cx + w * 1.6)),
                                  min(frame.shape[0], round(y + max(h * 2.5, w * 4))))
            self.status = 'Neck located - checking the whole bottle'
            return None
        if expected is not None:
            self.status = "AI re-detected bottle"
            return candidate
        movement = float("inf") if self.candidate is None else max(
            np.linalg.norm(candidate.cap - self.candidate.cap), np.linalg.norm(candidate.base - self.candidate.base))
        consistent = movement < max(12, np.linalg.norm(candidate.base - candidate.cap) * .10)
        if not consistent and self.candidate is not None:
            consistent = self._same_moving_bottle(frame, hand, candidate)
        self.count = self.count + 1 if consistent else 1
        self.candidate = candidate
        self._candidate_frame = frame
        self._candidate_hand = list(hand)
        self.status = f"Bottle located - confirming {min(self.count, self.stable_samples)}/{self.stable_samples}"
        return candidate if self.count >= self.stable_samples else None

    def acquire(self, frame, hand, now):
        candidate = self.update(frame, hand, now)
        if candidate is None:
            return None
        tracker = AxisBottleTracker(frame, candidate.cap, candidate.base, hand, candidate.body_width, candidate.source)
        tracker.endpoint_tracker.allow_geometry = False
        self.status = "Bottle located; contour measurement ready" if tracker.last.width_px is not None else "Bottle located; contour weak/hidden"
        self.reset()
        return tracker


class BottleSearchProcessor:
    """DNN ownership stays on a separate worker, away from hand inference."""

    def __init__(self, detector, model_name, shape_only=False):
        self.shape_only = shape_only
        if shape_only:
            from bottle_auto import AutomaticBottleFinder
            self.finder = AutomaticBottleFinder()
        else:
            self.finder = LearnedBottleFinder(detector)
        self.model_name = model_name
        self.generation = None

    def process(self, packet, commands):
        if commands or packet.generation != self.generation:
            self.finder.reset()
            self.finder.last_search = -float('inf')
            self.generation = packet.generation
        previous_search = self.finder.last_search
        candidate = self.finder.update(packet.frame, packet.hand, packet.captured_at, packet.expected)
        if self.finder.last_search == previous_search and (self.finder.detector is not None or self.shape_only):
            # A throttle-only tick must not erase an actual detection before
            # the hand worker has had a chance to consume it.
            return None
        return {'frame_id': packet.frame_id, 'captured_at': packet.captured_at, 'frame': packet.frame,
                'hand': packet.hand, 'generation': packet.generation, 'candidate': candidate,
                'detection': getattr(self.finder, 'detection', None), 'status': self.finder.status,
                'model_name': self.model_name}

    def close(self):
        pass


class AsyncBottleFinder:
    """Poll completed bottle searches without stalling per-frame hand tracking."""

    def __init__(self, factory):
        from realtime import LatestWorker
        self.worker = LatestWorker(factory, max_fps=2).start()
        self.status = 'Loading bottle model; hand tracking continues'
        self.model_name = 'loading'
        self.detection = None
        self.detection_time = -float('inf')
        self._request_id = self._seen_id = -1
        self._generation = 0

    def reset(self):
        self._generation += 1
        self.detection = None
        self.worker.command('reset')

    def update(self, frame, hand, now, expected=None):
        from types import SimpleNamespace
        self._request_id += 1
        # Images remain immutable in all workers; only trackers and canvases
        # are mutated. No queue of old images is accumulated.
        self.worker.submit(SimpleNamespace(frame_id=self._request_id, captured_at=now, frame=frame,
                                          hand=list(hand), expected=expected, generation=self._generation))
        if self.worker.error is not None:
            self.status = f'Bottle search failed: {self.worker.error}'
            self.model_name = 'unavailable'
            return None
        result = self.worker.read_latest()
        if result is None or result['frame_id'] == self._seen_id:
            return None
        self._seen_id = result['frame_id']
        self.model_name = result['model_name']
        if result['generation'] != self._generation or now - result['captured_at'] > 2.5:
            return None
        self.status = result['status']
        self.detection, self.detection_time = result['detection'], result['captured_at']
        candidate = result['candidate']
        if candidate is None:
            return None
        source_mask = hand_mask(frame.shape, result['hand'])
        current_mask = hand_mask(frame.shape, hand)
        signature = cap_identity(result['frame'], candidate.cap, candidate.body_width, source_mask)
        current = observe_endpoints(frame, candidate, current_mask, signature)
        if current is not None and same_foot(
                foot_patch(result['frame'], candidate.cap, candidate.base, candidate.body_width),
                foot_patch(frame, current.cap, current.base, current.body_width)):
            # Fresh cap/rim evidence breaks the deadlock where re-detection
            # required the same failed optical-flow bridge to succeed again.
            self.status = 'Bottle reobserved in current frame'
            return current
        # Coordinates from a slow DNN belong to its source image, not to the
        # current live frame. Transfer them with observed optical flow first.
        pose = BottlePose(candidate.cap, candidate.base)
        endpoint = EndpointTracker(result['frame'], pose, source_mask, candidate.body_width)
        endpoint.allow_geometry = False
        moved, _, _ = endpoint.track(frame, current_mask)
        if moved is None:
            self.status = 'Bottle detected; hold still briefly for alignment'
            return None
        scale = moved.length / pose.length
        return Landmarks(moved.cap, moved.base, candidate.body_width * scale, candidate.score, candidate.source)

    def acquire(self, frame, hand, now):
        candidate = self.update(frame, hand, now)
        if candidate is None:
            return None
        tracker = AxisBottleTracker(frame, candidate.cap, candidate.base, hand, candidate.body_width, candidate.source)
        tracker.endpoint_tracker.allow_geometry = False
        self.status = 'Bottle located; contour measurement ready' if tracker.last.width_px is not None else 'Bottle located; contour weak/hidden'
        return tracker

    def close(self):
        self.worker.close()
