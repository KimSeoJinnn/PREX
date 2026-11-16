"""Two-endpoint pose tracking and deformation in a bottle-aligned image.

Cap/base seeds are either clicked or proposed by a generic bottle box. There is
no trained cap/base keypoint detector. Only observed endpoint matches permit a
measurement; similarity alignment removes 2-D motion, not 3-D perspective.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from bottle_outline import OutlineMeasurement, edge_image, hand_mask
from bottle_occlusion import body_occlusion_mask
from bottle_landmarks import Landmarks, detect_landmarks
from bottle_endpoints import cap_identity, same_cap, observe_endpoints, foot_patch, same_foot
from bottle_joint_estimate import JointContourEstimator, JOINT_METRICS
from bottle_hand_tracking import GripHandTracker


CANONICAL_LENGTH = 320.0
CANONICAL_SIZE = (320, 384)
CAP_POSITION = np.array([160.0, 32.0])
STATIONS = np.linspace(0.27, 0.87, 31)
AXIS_METRICS = (
    "axis_length_px", "axis_tilt_deg", "axis_scale_ratio", "profile_coverage",
    "median_width_reduction_pct", "filtered_width_reduction_pct",
    "dent_p95_pct", "expansion_p95_pct", "left_dent_p95_pct", "right_dent_p95_pct",
    "visible_area_reduction_pct", "area_coverage", "contour_rms_pct",
    "center_shift_rms_pct", "baseline_noise_pct", "hand_occluded_profiles",
    "left_edge_coverage", "right_edge_coverage",
    *JOINT_METRICS,
)


@dataclass(frozen=True)
class BottlePose:
    cap: np.ndarray
    base: np.ndarray

    @property
    def length(self):
        return float(np.linalg.norm(self.base - self.cap))

    @property
    def direction(self):
        return (self.base - self.cap) / self.length

    @property
    def normal(self):
        direction = self.direction
        return np.array([direction[1], -direction[0]])

    @property
    def matrix(self):
        rotation = np.stack((self.normal, self.direction)) * (CANONICAL_LENGTH / self.length)
        return np.column_stack((rotation, CAP_POSITION - rotation @ self.cap)).astype(np.float32)

    def rectify(self, image, nearest=False):
        return cv2.warpAffine(image, self.matrix, CANONICAL_SIZE,
                              flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    def image_points(self, points):
        inverse = cv2.invertAffineTransform(self.matrix)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(points, inverse).reshape(-1, 2)

    @property
    def points(self):
        return tuple(tuple(float(v) for v in point) for point in (self.cap, self.base))


def propose_endpoints(frame, detection, hand):
    """Image heuristics refine a box proposal; callers must allow correction."""
    x, y, width, height = detection.region
    patch = frame[y:y + height, x:x + width]
    if patch.size == 0:
        raise ValueError("Empty bottle region")
    cap = np.array([x + width / 2, y + height * .07])
    base = np.array([x + width / 2, y + height * .96])
    blocked = hand_mask(frame.shape, hand)[y:y + height, x:x + width]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    colored[round(height * .25):] = 0
    colored[blocked > 0] = 0
    contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        cx, cy, cw, ch = cv2.boundingRect(contour)
        center_offset = abs(cx + cw / 2 - width / 2) / width
        # Saturated background strips can have more area than a colored cap.
        # A cap must be near the neck, compact, and horizontally plausible.
        if (area >= 25 and cw >= max(8, width * .12) and ch >= 5 and cw < width * .9
                and .7 <= cw / ch <= 4.5 and center_offset < .32
                and cy + ch / 2 < height * .18):
            candidates.append((area / ((1 + cy / max(height, 1)) * (1 + 6 * center_offset)), contour))
    refined_cap = False
    if candidates:
        contour = max(candidates, key=lambda item: item[0])[1]
        moments = cv2.moments(contour)
        cap = np.array([x + moments["m10"] / moments["m00"], y + moments["m01"] / moments["m00"]])
        refined_cap = True
    gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude[:round(height * .89)] = 0
    magnitude[blocked > 0] = 0
    threshold = max(15, float(np.max(magnitude)) * .18)
    yy, xx = np.nonzero(magnitude >= threshold)
    if len(xx) >= 20:
        # Paired rim extent is more stable than its brightest individual lobe.
        lo, hi = np.percentile(xx, [5, 95])
        base = np.array([x + (lo + hi) / 2, y + float(np.median(yy))])
    return cap, base, "detector_image_proposal" if refined_cap else "detector_box_proposal"


class EndpointTracker:
    """Forward/backward LK, two supported feature clusters, template checks."""

    def __init__(self, frame, pose, blocked, body_width=None):
        self.pose = self.initial_pose = pose
        self.previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference = pose.rectify(self.previous)
        self.body_ratio = (body_width or pose.length * .37) / pose.length
        self.cap_signature = cap_identity(frame, pose.cap, self.body_ratio * pose.length, blocked)
        self.foot_reference = foot_patch(frame, pose.cap, pose.base, self.body_ratio * pose.length)
        self.features, self.groups = self._features(self.previous, pose, blocked)
        self.quality = 1.0
        self._refresh_age = 0
        self.reference_body_ratio = None
        self.allow_geometry = True
        self.allow_local_endpoints = False
        self.local_recoveries = 0

    def _recover_geometry(self, frame, blocked):
        """Reobserve landmarks, rather than extrapolate a missing optical flow."""
        if self.allow_local_endpoints:
            seed = Landmarks(self.pose.cap, self.pose.base, self.body_ratio * self.pose.length,
                             .95, 'learned_tracking_seed')
            candidate = observe_endpoints(frame, seed, blocked, self.cap_signature)
            if candidate is not None and same_foot(self.foot_reference, foot_patch(
                    frame, candidate.cap, candidate.base, self.body_ratio * np.linalg.norm(candidate.base - candidate.cap))):
                result = self.adopt_geometry(frame, blocked, candidate)
                if result is not None:
                    return result
        if not self.allow_geometry:
            return None
        minimum = np.minimum(self.pose.cap, self.pose.base)
        maximum = np.maximum(self.pose.cap, self.pose.base)
        length = self.pose.length
        region = (minimum[0] - length * .35, minimum[1] - length * .12,
                  maximum[0] - minimum[0] + length * .70,
                  maximum[1] - minimum[1] + length * .27)
        found = detect_landmarks(frame, [], region, (self.pose.cap, self.pose.base), blocked)
        if len(found) != 1:
            return None
        return self.adopt_geometry(frame, blocked, found[0])

    def adopt_geometry(self, frame, blocked, candidate):
        pose = BottlePose(candidate.cap, candidate.base)
        movement = max(np.linalg.norm(pose.cap - self.pose.cap), np.linalg.norm(pose.base - self.pose.base)) / self.pose.length
        learned = candidate.source.startswith('learned_')
        observed = candidate.endpoint_quality is not None and min(candidate.endpoint_quality) >= .60
        if movement > (.60 if learned else .28):
            return None
        if not .80 <= pose.length / self.pose.length <= 1.25:
            return None
        ratio = candidate.body_width / pose.length
        if not observed and self.reference_body_ratio is not None and not .72 <= ratio / self.reference_body_ratio <= 1.30:
            return None
        # Uniform white caps can have too little texture for normalized NCC.
        # Require a matching appearance as well as cap/body/terminating-rim evidence.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aligned = pose.rectify(gray)
        differences = []
        for center_y in (32, 352):
            current = aligned[center_y - 10:center_y + 11, 150:171]
            reference = self.reference[center_y - 10:center_y + 11, 150:171]
            differences.append(abs(float(np.mean(current)) - float(np.mean(reference))))
        minimum_score = .25 if candidate.source.startswith("learned_") else .72
        if observed:
            current_cap = cap_identity(frame, pose.cap, self.body_ratio * pose.length, blocked)
            if candidate.score < .45 or not same_cap(self.cap_signature, current_cap):
                return None
        else:
            if max(differences) > 45 or candidate.score < minimum_score:
                return None
            if movement > .28 and (candidate.score < .45 or self._verify(gray, pose) < .60):
                return None
        self.pose = pose
        self.previous = gray
        self.features, self.groups = self._features(gray, pose, blocked)
        self.quality = candidate.score * (1 - max(differences) / 100)
        if observed:
            # Refresh only the motion/appearance template after direct image
            # evidence. AxisBottleTracker's deformation baseline is untouched.
            # A former neck grip/background must not veto every future frame.
            self.reference = aligned
            self.foot_reference = foot_patch(frame, pose.cap, pose.base, self.body_ratio * pose.length)
            self.quality = min(candidate.score, *candidate.endpoint_quality)
            self.local_recoveries += 1
        self._refresh_age = 0
        return pose, self.quality, "Axis re-detected from cap/base"

    def _features(self, gray, pose, blocked):
        # Transparent interiors show stationary background corners. A large
        # disk around the base lets those corners outvote the moving bottle.
        # Use the compact cap and the observed terminating rim, not the disk.
        aligned = pose.rectify(gray)
        excluded = pose.rectify(blocked, nearest=True)
        excluded[pose.rectify(np.full(gray.shape, 255, np.uint8), nearest=True) == 0] = 255
        width = np.clip(self.body_ratio * CANONICAL_LENGTH, 35, 230)
        masks = [np.zeros_like(aligned), np.zeros_like(aligned)]
        half_cap = max(9, round(width * .30))
        masks[0][14:51, 160 - half_cap:161 + half_cap] = 255
        smooth = cv2.GaussianBlur(aligned, (3, 3), 0)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
        gy[np.abs(gx) > np.abs(gy) * 1.6] = 0
        lo, hi = round(160 - width * .48), round(160 + width * .48) + 1
        rim = trace_contour(gy[:, lo:hi].T, excluded[:, lo:hi].T,
                            np.full(hi - lo, 352.), np.full(hi - lo, 17.))
        for column in np.flatnonzero(np.isfinite(rim)):
            row = round(rim[column])
            masks[1][max(0, row - 3):row + 4, lo + column] = 255
        features, groups = [], []
        inverse = cv2.invertAffineTransform(pose.matrix)
        for group, (point, fraction) in enumerate(((pose.cap, .12), (pose.base, .21))):
            x, y = np.rint(point).astype(int)
            radius = max(12, round(pose.length * fraction))
            x0, y0 = max(0, x - radius), max(0, y - radius)
            patch = gray[y0:min(gray.shape[0], y + radius + 1), x0:min(gray.shape[1], x + radius + 1)]
            support = cv2.warpAffine(masks[group], inverse, gray.shape[1::-1], flags=cv2.INTER_NEAREST)
            mask = support[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]].copy()
            mask[blocked[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] > 0] = 0
            points = cv2.goodFeaturesToTrack(patch, 32, .015, 4, mask=mask, blockSize=5)
            if points is not None:
                points += np.array([x0, y0], np.float32)
                features.extend(points.reshape(-1, 2))
                groups.extend([group] * len(points))
        return np.asarray(features, np.float32).reshape(-1, 1, 2), np.asarray(groups)

    def _verify(self, gray, pose):
        aligned = pose.rectify(gray)
        scores = []
        for center_y, radius in ((32, 26), (352, 30)):
            ref = self.reference[center_y - radius:center_y + radius + 1, 160 - radius:161 + radius]
            search = aligned[center_y - radius - 3:center_y + radius + 4, 160 - radius - 3:164 + radius]
            if search.shape[0] < ref.shape[0] or search.shape[1] < ref.shape[1] or np.std(ref) < 2:
                return 0.0
            scores.append(float(cv2.minMaxLoc(cv2.matchTemplate(search, ref, cv2.TM_CCOEFF_NORMED))[1]))
        return min(scores)

    def _template_fallback(self, gray):
        points = []
        for point, fraction in ((self.pose.cap, .12), (self.pose.base, .17)):
            radius = max(12, round(self.pose.length * fraction))
            size = 2 * radius + 1
            template = cv2.getRectSubPix(self.previous, (size, size), tuple(point.astype(float)))
            if np.std(template) < 3:
                return None
            travel = max(18, round(self.pose.length * .08))
            x, y = np.rint(point).astype(int)
            x0, y0 = max(0, x - radius - travel), max(0, y - radius - travel)
            patch = gray[y0:min(gray.shape[0], y + radius + travel + 1),
                         x0:min(gray.shape[1], x + radius + travel + 1)]
            if patch.shape[0] < size or patch.shape[1] < size:
                return None
            scores = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            _, best, _, position = cv2.minMaxLoc(scores)
            px, py = position
            competing = scores.copy()
            competing[max(0, py - 5):py + 6, max(0, px - 5):px + 6] = -1
            if best < .78 or best - float(competing.max()) < .025:
                return None
            points.append(np.array([x0 + px + radius, y0 + py + radius], float))
        return BottlePose(*points)

    def track(self, frame, blocked):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for point in (self.pose.cap, self.pose.base):
            x, y = np.rint(point).astype(int)
            patch = blocked[max(0, y - 8):y + 9, max(0, x - 8):x + 9]
            if patch.size == 0 or np.mean(patch > 0) > .35:
                recovered = self._recover_geometry(frame, blocked)
                if recovered is not None:
                    return recovered
                return None, 0.0, "Cap/base hidden - P to reselect"
        if np.array_equal(gray, self.previous):
            return self.pose, self.quality, "Axis OK"
        pose = None
        cap_motion = None
        retained = retained_groups = None
        support = 0.0
        if len(self.features) >= 6:
            forward, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous, gray, self.features, None, winSize=(25, 25), maxLevel=3)
            if forward is not None:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous, forward, None, winSize=(25, 25), maxLevel=3)
                if backward is not None:
                    good = (status.ravel() > 0) & (back_status.ravel() > 0)
                    good &= np.linalg.norm(backward - self.features, axis=2).ravel() < 1.3
                    good &= errors.ravel() < 35
                    xy = forward.reshape(-1, 2)
                    inside = (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
                    good &= inside
                    for index in np.flatnonzero(good):
                        x, y = np.rint(xy[index]).astype(int)
                        if blocked[min(y, gray.shape[0] - 1), min(x, gray.shape[1] - 1)]:
                            good[index] = False
                    if np.count_nonzero(good) >= 6:
                        cap_good = good & (self.groups == 0)
                        if np.count_nonzero(cap_good) >= 8:
                            cap_points = self.features[cap_good].reshape(-1, 2)
                            # A compact 2-D patch supports cap rotation. A row
                            # of points or a tiny cluster does not.
                            if np.min(np.ptp(cap_points, axis=0)) >= 10:
                                cap_transform, cap_inliers = cv2.estimateAffinePartial2D(
                                    self.features[cap_good], forward[cap_good], method=cv2.RANSAC,
                                    ransacReprojThreshold=.65)
                                if cap_transform is not None and cap_inliers is not None and np.mean(cap_inliers) >= .75:
                                    cap_motion = cv2.transform(np.array([[self.pose.cap, self.pose.base]], np.float32),
                                                               cap_transform)[0]
                        transform, inliers = cv2.estimateAffinePartial2D(
                            self.features[good], forward[good], method=cv2.RANSAC,
                            ransacReprojThreshold=1.8)
                        if transform is not None and inliers is not None:
                            valid_groups = self.groups[good][inliers.ravel() > 0]
                            if all(np.count_nonzero(valid_groups == group) >= 3 for group in (0, 1)):
                                moved = cv2.transform(np.array([[self.pose.cap, self.pose.base]], np.float32), transform)[0]
                                pose = BottlePose(moved[0].astype(float), moved[1].astype(float))
                                support = float(np.mean(inliers))
                                retained = forward[good][inliers.ravel() > 0]
                                retained_groups = self.groups[good][inliers.ravel() > 0]
        if pose is None:
            pose = self._template_fallback(gray)
            support = .75
        if pose is None or pose.length < 60:
            recovered = self._recover_geometry(frame, blocked)
            if recovered is not None:
                return recovered
            return None, 0.0, "Axis lost - show cap/base or press P"
        if cap_motion is not None and max(np.linalg.norm(pose.cap - cap_motion[0]),
                                          np.linalg.norm(pose.base - cap_motion[1])) > max(3., self.pose.length * .01):
            # Do not report deformation when base/background tracks contradict
            # the independently observed cap rotation. Re-detect instead of
            # extrapolating the hidden/unsupported base from the cap alone.
            recovered = self._recover_geometry(frame, blocked)
            if recovered is not None:
                return recovered
            return None, 0.0, "Endpoint motion disagrees; re-detecting bottle"
        step_scale = pose.length / self.pose.length
        total_scale = pose.length / self.initial_pose.length
        old, new = self.pose.direction, pose.direction
        cross = old[0] * new[1] - old[1] * new[0]
        angle = abs(math.degrees(math.atan2(float(cross), float(old @ new))))
        if not .82 < step_scale < 1.22 or not .5 < total_scale < 2 or angle > 20:
            return None, 0.0, "Axis motion too large - P to reselect"
        for point in (pose.cap, pose.base):
            if np.any(point < 5) or point[0] >= gray.shape[1] - 5 or point[1] >= gray.shape[0] - 5:
                return None, 0.0, "Keep cap and base inside frame"
            x, y = np.rint(point).astype(int)
            if np.mean(blocked[max(0, y - 8):y + 9, max(0, x - 8):x + 9] > 0) > .35:
                return None, 0.0, "Cap/base hidden - P to reselect"
        agreement = self._verify(gray, pose)
        if agreement < .50:
            recovered = self._recover_geometry(frame, blocked)
            if recovered is not None:
                return recovered
            return None, 0.0, "Cap/base appearance mismatch - P retry"
        self.pose = pose
        self.previous = gray
        self._refresh_age += 1
        if (retained is None or self._refresh_age >= 10
                or any(np.count_nonzero(retained_groups == group) < 8 for group in (0, 1))):
            self.features, self.groups = self._features(gray, pose, blocked)
            self._refresh_age = 0
        else:
            self.features, self.groups = retained, retained_groups
        self.quality = min(support, agreement)
        return pose, self.quality, "Axis OK"


def trace_contour(gradient, blocked, expected, radius, polarity=None):
    """Trace visible runs independently; masked or weak edges stay missing."""
    height, width = gradient.shape
    if height == 0:
        return np.empty(0, dtype=float)
    # Most of the rectified image is outside this side's search window. Avoid
    # evaluating every background column in the dynamic program.
    start = max(0, int(np.floor(np.min(expected - radius))) - 5)
    end = min(width, int(np.ceil(np.max(expected + radius))) + 6)
    if 0 < end - start < width - 8:
        return trace_contour(gradient[:, start:end], blocked[:, start:end],
                             expected - start, radius, polarity) + start
    columns = np.arange(width)[None, :]
    signal = np.abs(gradient) if polarity is None else np.maximum(0, gradient * polarity[:, None])
    signal = signal.copy()
    signal[blocked > 0] = 0
    allowed = np.abs(columns - expected[:, None]) <= radius[:, None]
    observable = np.any(allowed & (blocked == 0), axis=1)
    if not np.all(observable):
        # A hidden row has no edge evidence. Solve the visible runs separately
        # instead of allowing an arbitrary path THROUGH the hand to pull the
        # contours on either side of it. Fully hidden rows also cost no DP time.
        result = np.full(height, np.nan)
        transitions = np.diff(np.r_[False, observable, False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
            result[lo:hi] = trace_contour(
                gradient[lo:hi], blocked[lo:hi], expected[lo:hi], radius[lo:hi],
                None if polarity is None else polarity[lo:hi])
        return result
    # Saturating edge rewards stop a bright internal reflection dominating.
    cost = -np.minimum(signal / 30.0, 2.5)
    cost += .65 * ((columns - expected[:, None]) / np.maximum(radius[:, None], 4)) ** 2
    cost[~allowed] = 1000
    previous = cost[0].copy()
    parents = np.zeros((height, width), np.int8)
    steps = np.arange(-4, 5)
    neighbors = np.arange(width)[:, None] + steps
    inside = (neighbors >= 0) & (neighbors < width)
    neighbors = np.clip(neighbors, 0, width - 1)
    penalty = .16 * np.abs(steps)
    for row in range(1, height):
        choices = previous[neighbors] + penalty
        choices[~inside] = 1e6
        best = np.argmin(choices, axis=1)
        parents[row] = steps[best]
        previous = cost[row] + choices[np.arange(width), best]
    path = np.zeros(height, np.int32)
    path[-1] = int(np.argmin(previous))
    for row in range(height - 1, 0, -1):
        path[row - 1] = path[row] + parents[row, path[row]]
    rows = np.arange(height)
    valid = allowed[rows, path] & (blocked[rows, path] == 0) & (signal[rows, path] >= 12)
    result = path.astype(float)
    # Subpixel gradient-peak refinement, limited to the chosen path's lobe.
    interior = (path > 0) & (path < width - 1) & valid
    rr, xx = rows[interior], path[interior]
    a, b, c = signal[rr, xx - 1], signal[rr, xx], signal[rr, xx + 1]
    denominator = a - 2 * b + c
    offset = np.divide(.5 * (a - c), denominator, out=np.zeros_like(a), where=np.abs(denominator) > 1e-6)
    result[interior] += np.clip(offset, -.5, .5)
    result[~valid] = np.nan
    return result


class AxisBottleTracker:
    def __init__(self, frame, cap, base, hand, width_hint=None, source="manual"):
        cap, base = np.asarray(cap, float), np.asarray(base, float)
        pose = BottlePose(cap, base)
        height, width = frame.shape[:2]
        if not np.isfinite([cap, base]).all() or pose.length < 80:
            raise ValueError("Cap and base must be finite and at least 80 pixels apart")
        if any(np.any(point < 6) or point[0] >= width - 6 or point[1] >= height - 6 for point in (cap, base)):
            raise ValueError("Select cap/base centers inside the image")
        self.pose = self.initial_pose = pose
        self.source = source
        self.endpoint_tracker = EndpointTracker(frame, pose, hand_mask(frame.shape, hand), width_hint)
        self.endpoint_tracker.allow_local_endpoints = source.startswith('learned_')
        self.positions = STATIONS.copy()
        self.reference_left = self.reference_right = None
        self.baseline_left = self.baseline_right = None
        self.baseline_noise = None
        self.last = OutlineMeasurement(reason="Axis selected; checking body edges")
        self._filtered = self._last_time = None
        self._seed_width = width_hint * CANONICAL_LENGTH / pose.length if width_hint else None
        self.reference_width = 0.0
        self.body_mask = np.zeros(frame.shape[:2], np.uint8)
        self._seen_hand = False
        self._unmasked_reference_image = None
        self.reference_revision = 0
        self._occluded_profiles = np.zeros(len(self.positions), bool)
        self.visible_left = np.full(len(self.positions), np.nan)
        self.visible_right = np.full(len(self.positions), np.nan)
        self.outline_segments = []
        self.body_outline_segments = []
        self._previous_left = self._previous_right = None
        self._polarity_left = self._polarity_right = None
        self.joint_estimator = JointContourEstimator(32 + self.positions * CANONICAL_LENGTH,
                                                    self.initial_pose.length / CANONICAL_LENGTH,
                                                    self._seed_width or self.endpoint_tracker.body_ratio * CANONICAL_LENGTH)
        self.joint_hand = GripHandTracker()
        self.measure(frame, hand)

    @classmethod
    def from_detection(cls, frame, detection, hand):
        x, y, width, height = detection.region
        try:
            cap, base, source = propose_endpoints(frame, detection, hand)
            length = float(np.linalg.norm(base - cap))
            # Remove the axis's horizontal extent from a tilted object's box.
            cosine = abs((base - cap)[1]) / max(length, 1)
            hint = max(length * .16, (width - abs(base[0] - cap[0])) / max(.5, cosine))
            return cls(frame, cap, base, hand, width_hint=hint, source=source)
        except ValueError:
            return None

    @property
    def profile_count(self):
        return len(self.positions)

    @property
    def region(self):
        points = self.pose.image_points([[25, 110], [295, 110], [295, 320], [25, 320]])
        x, y, width, height = cv2.boundingRect(points.astype(np.float32))
        return x, y, width, height

    def _width_seed(self, strength):
        if self._seed_width is not None:
            return float(np.clip(self._seed_width, 35, 240))
        presence = np.mean(np.abs(strength) >= 12, axis=0)
        peaks = np.flatnonzero((presence >= .4) & (np.abs(np.arange(strength.shape[1]) - 160) >= 24))
        left = peaks[(peaks > 35) & (peaks < 136)]
        right = peaks[(peaks > 184) & (peaks < 285)]
        if len(left) and len(right):
            return float(right[-1] - left[0])
        return CANONICAL_LENGTH * .32

    @staticmethod
    def _tuple(array):
        return tuple(float(value) if np.isfinite(value) else None for value in array)

    def _extract(self, frame, blocked, pose):
        aligned = pose.rectify(frame)
        invalid = pose.rectify(np.full(frame.shape[:2], 255, np.uint8), nearest=True) == 0
        hand_aligned = pose.rectify(blocked, nearest=True)
        mask = hand_aligned.copy()
        mask[invalid] = 255
        strength = edge_image(aligned)
        self._extend_visible_outline(strength, mask, pose)
        start = round(32 + STATIONS[0] * CANONICAL_LENGTH) - 3
        end = round(32 + STATIONS[-1] * CANONICAL_LENGTH) + 4
        gradient, mask = strength[start:end], mask[start:end]
        local_hand = hand_aligned[start:end]
        rows = np.arange(start, end)
        station_rows = 32 + self.positions * CANONICAL_LENGTH
        if self.reference_left is None:
            width = self._width_seed(np.where(mask > 0, 0, gradient))
            left = np.full(len(rows), 160 - width / 2)
            right = np.full(len(rows), 160 + width / 2)
            radius = np.full(len(rows), max(12, width * .23))
            polarity_left = polarity_right = None
        else:
            good = np.isfinite(self.reference_left) & np.isfinite(self.reference_right)
            reference_widths = self.reference_right - self.reference_left
            if np.count_nonzero(good) < 6:
                return None
            left = np.interp(rows, station_rows[good], self._previous_left[good])
            right = np.interp(rows, station_rows[good], self._previous_right[good])
            radius = np.interp(rows, station_rows[good], np.maximum(12, reference_widths[good] * .35))
            polarity_left = np.interp(rows, station_rows[good], self._polarity_left[good])
            polarity_right = np.interp(rows, station_rows[good], self._polarity_right[good])
            polarity_left = np.where(polarity_left >= 0, 1, -1)
            polarity_right = np.where(polarity_right >= 0, 1, -1)
        # An occluded expected edge is missing, not permission to jump outward
        # to the finger's edge or a background line just beyond the hand mask.
        rr = np.arange(len(rows))[:, None]
        margin = max(3, min(8, round(float(np.median(right - left)) * .04)))
        hand_rows = []
        side_masks = []
        for expected in (left, right):
            xx = np.rint(expected).astype(int)[:, None] + np.arange(-margin, margin + 1)
            guard = np.any(mask[rr, np.clip(xx, 0, mask.shape[1] - 1)] > 0, axis=1)
            hand_rows.append(np.any(local_hand[rr, np.clip(xx, 0, mask.shape[1] - 1)] > 0, axis=1))
            side_mask = mask.copy()
            side_mask[guard] = 255
            side_masks.append(side_mask)
        path_left = trace_contour(gradient, side_masks[0], left, radius, polarity_left)
        path_right = trace_contour(gradient, side_masks[1], right, radius, polarity_right)
        # Display the observed per-row paths, not just the 31 width stations.
        # Short visible gaps between fingers can miss every station center.
        # These additional display samples never enter measurement/baseline.
        self.body_outline_segments = []
        for path in (path_left, path_right):
            finite = np.isfinite(path)
            transitions = np.diff(np.r_[False, finite, False].astype(np.int8))
            for lo, hi in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                if hi - lo >= 3:
                    self.body_outline_segments.append(pose.image_points(np.column_stack((path[lo:hi], rows[lo:hi]))))
        indices = np.rint(station_rows).astype(int)[:, None] - start + np.arange(-3, 4)
        l, r = path_left[indices], path_right[indices]
        both = np.isfinite(l) & np.isfinite(r)
        visible = []
        for path, hand_guard in zip((l, r), hand_rows):
            # Four observed rows and an unoccluded center row are enough to
            # display this side. A hidden neighbor must not erase the other
            # visible side of the bottle.
            enough_side = (np.count_nonzero(np.isfinite(path), axis=1) >= 4) & ~hand_guard[indices[:, 3]]
            values = np.full(len(station_rows), np.nan)
            if np.any(enough_side):
                values[enough_side] = np.nanmedian(path[enough_side], axis=1)
            visible.append(values)
        self.visible_left, self.visible_right = visible
        self._occluded_profiles = ((np.any(hand_rows[0][indices], axis=1) & ~np.isfinite(visible[0]))
                                   | (np.any(hand_rows[1][indices], axis=1) & ~np.isfinite(visible[1])))
        enough = (np.count_nonzero(both, axis=1) >= 4) & np.isfinite(visible[0]) & np.isfinite(visible[1])
        profile_left = np.full(len(station_rows), np.nan)
        profile_right = profile_left.copy()
        signs_left, signs_right = np.ones(len(station_rows)), np.ones(len(station_rows))
        if np.any(enough):
            profile_left[enough] = np.nanmedian(np.where(both[enough], l[enough], np.nan), axis=1)
            profile_right[enough] = np.nanmedian(np.where(both[enough], r[enough], np.nan), axis=1)
            for path, signs in ((l, signs_left), (r, signs_right)):
                xx = np.rint(np.where(both[enough], path[enough], 0)).astype(int)
                values = gradient[indices[enough], xx]
                median = np.nanmedian(np.where(both[enough], values, np.nan), axis=1)
                signs[enough] = np.where(median >= 0, 1, -1)
        valid = (profile_right - profile_left >= 20) & (profile_left < 185) & (profile_right > 135)
        profile_left[~valid] = profile_right[~valid] = np.nan
        return profile_left, profile_right, signs_left, signs_right

    def _extend_visible_outline(self, strength, mask, pose):
        """Observe neck/base edges for display and explicit proxy anchors.

        No extrapolated path, missing sample, or extension enters a width,
        baseline, area, or deformation calculation.
        """
        self.outline_segments = []
        width = self.reference_width * CANONICAL_LENGTH / self.initial_pose.length if self.reference_width else self._width_seed(strength)
        for lo, hi in ((.04, .25), (.84, .995)):
            start, end = round(32 + lo * CANONICAL_LENGTH), round(32 + hi * CANONICAL_LENGTH)
            rows = np.arange(start, end)
            fraction = (rows - 32) / CANONICAL_LENGTH
            taper = np.interp(fraction, (.04, .10, .24, .90, .98), (.45, .60, 1., 1., .80))
            radius = np.full(len(rows), max(12., width * .23))
            for sign in (-1, 1):
                expected = 160 + sign * width * taper / 2
                local_mask = mask[start:end].copy()
                xx = np.rint(expected).astype(int)[:, None] + np.arange(-3, 4)
                hidden = np.any(local_mask[np.arange(len(rows))[:, None], np.clip(xx, 0, mask.shape[1] - 1)] > 0, axis=1)
                local_mask[hidden] = 255
                path = trace_contour(strength[start:end], local_mask, expected, radius)
                segment = []
                for index in range(0, len(rows), 3):
                    if not np.isfinite(path[index]):
                        if len(segment) >= 3:
                            self.outline_segments.append(pose.image_points(segment))
                        segment = []
                    else:
                        segment.append((path[index], rows[index]))
                if len(segment) >= 3:
                    self.outline_segments.append(pose.image_points(segment))

    def set_baseline(self, samples):
        suitable = [sample for sample in samples if len(sample.profile_left) == len(self.positions)]
        if len(suitable) < 3:
            self.baseline_left = self.baseline_right = None
            return False
        left = np.array([[np.nan if v is None else v for v in sample.profile_left] for sample in suitable])
        right = np.array([[np.nan if v is None else v for v in sample.profile_right] for sample in suitable])
        self.baseline_left = np.full(len(self.positions), np.nan)
        self.baseline_right = np.full(len(self.positions), np.nan)
        noise = []
        for index in range(len(self.positions)):
            valid = np.isfinite(left[:, index]) & np.isfinite(right[:, index])
            if np.count_nonzero(valid) < max(3, math.ceil(len(suitable) * .6)):
                continue
            self.baseline_left[index] = np.median(left[valid, index])
            self.baseline_right[index] = np.median(right[valid, index])
            widths = right[valid, index] - left[valid, index]
            noise.append(100 * 1.4826 * np.median(np.abs(widths - np.median(widths))) / np.median(widths))
        self.baseline_noise = float(np.median(noise)) if noise else None
        self._filtered = self._last_time = None
        return len(noise) >= 6

    def reset_baseline(self):
        self.baseline_left = self.baseline_right = None
        self.baseline_noise = None
        self._filtered = self._last_time = None
        self.joint_estimator.reset()

    def _deformation(self, left, right, timestamp):
        metrics = {name: None for name in AXIS_METRICS}
        metrics.update(axis_length_px=self.pose.length,
                       axis_tilt_deg=math.degrees(math.atan2(self.pose.direction[0], self.pose.direction[1])),
                       axis_scale_ratio=self.pose.length / self.initial_pose.length)
        valid = np.isfinite(left) & np.isfinite(right)
        metrics["profile_coverage"] = float(np.mean(valid))
        if self.baseline_left is None:
            return metrics
        reference_widths = self.baseline_right - self.baseline_left
        valid &= np.isfinite(reference_widths) & (reference_widths > 0)
        metrics["profile_coverage"] = float(np.mean(valid))
        metrics["baseline_noise_pct"] = self.baseline_noise
        intervals = valid[:-1] & valid[1:]
        metrics["area_coverage"] = float(np.mean(intervals))
        if np.count_nonzero(valid) < 6:
            self._filtered = self._last_time = None
            return metrics
        widths = right - left
        reductions = 100 * (1 - widths[valid] / reference_widths[valid])
        left_dent = 100 * (left[valid] - self.baseline_left[valid]) / reference_widths[valid]
        right_dent = 100 * (self.baseline_right[valid] - right[valid]) / reference_widths[valid]
        median = float(np.median(reductions))
        metrics.update(median_width_reduction_pct=median,
                       dent_p95_pct=float(np.percentile(np.maximum(0, reductions), 95)),
                       expansion_p95_pct=float(np.percentile(np.maximum(0, -reductions), 95)),
                       left_dent_p95_pct=float(np.percentile(np.maximum(0, left_dent), 95)),
                       right_dent_p95_pct=float(np.percentile(np.maximum(0, right_dent), 95)),
                       contour_rms_pct=float(np.sqrt(np.mean((left_dent ** 2 + right_dent ** 2) / 2))),
                       center_shift_rms_pct=float(np.sqrt(np.mean(((left_dent - right_dent) / 2) ** 2))))
        if np.mean(intervals) >= .6:
            current_area = np.sum((widths[:-1] + widths[1:])[intervals])
            reference_area = np.sum((reference_widths[:-1] + reference_widths[1:])[intervals])
            metrics["visible_area_reduction_pct"] = float(100 * (1 - current_area / reference_area))
        if timestamp is None or self._last_time is None or timestamp <= self._last_time:
            self._filtered = median
        else:
            alpha = 1 - math.exp(-(timestamp - self._last_time) / .12)
            self._filtered += alpha * (median - self._filtered)
        self._last_time = timestamp
        metrics["filtered_width_reduction_pct"] = self._filtered
        return metrics

    def _joint_only(self, frame, pose, axis_quality, timestamp, track, reason):
        """Keep a short image track visible while all unsupported readings clear."""
        self._filtered = self._last_time = None
        empty = np.full(len(self.positions), np.nan)
        metrics = self._deformation(empty, empty, timestamp)
        if len(track.points) == 21:
            self.body_mask = body_occlusion_mask(frame, track.points)
            self.joint_estimator.update(pose, track.points, [], self.body_mask, timestamp,
                                        hand_side=track.side, display_only=track.display_only)
        else:
            self.joint_estimator.missing(timestamp, 'Estimate paused: hand lost')
        metrics.update(self.joint_estimator.metrics)
        self.last = OutlineMeasurement(reason=reason, axis_points=pose.points, axis_quality=axis_quality,
                                       band_widths=(None,) * len(self.positions),
                                       profile_left=(None,) * len(self.positions),
                                       profile_right=(None,) * len(self.positions), metrics=metrics,
                                       joint_estimated_left=self._tuple(self.joint_estimator.profile_left),
                                       joint_estimated_right=self._tuple(self.joint_estimator.profile_right))
        return self.last

    def measure(self, frame, hand, timestamp=None, hand_side=None, hand_angles=None):
        self.joint_estimator.clear_current()
        self.visible_left[:] = np.nan
        self.visible_right[:] = np.nan
        self.outline_segments = []
        self.body_outline_segments = []
        blocked = hand_mask(frame.shape, hand)
        pose, axis_quality, reason = self.endpoint_tracker.track(frame, blocked)
        if pose is None:
            self.joint_hand.reset()
            self.joint_estimator.missing(timestamp, 'Estimate paused: bottle axis lost')
            self._filtered = self._last_time = None
            self.last = OutlineMeasurement(reason=reason, band_widths=(None,) * len(self.positions))
            return self.last
        self.pose = pose
        track = self.joint_hand.update(frame, hand, hand_side, timestamp)
        if len(hand) == 21:
            if not self._seen_hand and self._unmasked_reference_image is not None:
                # A delayed hand detection must not preserve a reference made
                # from the same already-present hand's silhouette. Compare in
                # bottle coordinates, restricted to the actual body/hand area.
                inside = pose.rectify(blocked, nearest=True) > 0
                half = max(20, round((self._seed_width or 120) * .60))
                body_region = np.zeros_like(inside)
                body_region[110:320, max(0, 160 - half):min(320, 161 + half)] = True
                inside &= body_region
                if np.count_nonzero(inside) >= 100:
                    current = pose.rectify(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    difference = np.abs(current.astype(float) - self._unmasked_reference_image)
                    if float(np.median(difference[inside])) < 12:
                        self.reference_left = self.reference_right = None
                        self._previous_left = self._previous_right = None
                        self._polarity_left = self._polarity_right = None
                        self.reference_width = 0.
                        self.endpoint_tracker.reference_body_ratio = None
                        self.reset_baseline()
                        self.reference_revision += 1
                self._unmasked_reference_image = None
            self._seen_hand = True
        elif self._seen_hand:
            # A transient hand-detector miss must not suddenly make the hand
            # outline eligible for bottle calibration or deformation.
            return self._joint_only(frame, pose, axis_quality, timestamp, track,
                                    'Hand lost; body measurement paused')
        if track.display_only:
            return self._joint_only(frame, pose, axis_quality, timestamp, track,
                                    'Hand landmarks unstable; readings paused')
        self.body_mask = body_occlusion_mask(frame, hand, blocked)
        extracted = self._extract(frame, self.body_mask, pose)
        if extracted is None:
            return self._joint_only(frame, pose, axis_quality, timestamp, track,
                                    'Body edges weak; joint proxy only')
        left, right, signs_left, signs_right = extracted
        valid = np.isfinite(left) & np.isfinite(right)
        if self.reference_left is None and np.count_nonzero(valid) >= 6:
            # Refine with locked polarity before storing the reference curve.
            self.reference_left, self.reference_right = left.copy(), right.copy()
            self._previous_left, self._previous_right = left.copy(), right.copy()
            self._polarity_left, self._polarity_right = signs_left, signs_right
            refined = self._extract(frame, self.body_mask, pose)
            if refined is not None:
                left, right = refined[:2]
                valid = np.isfinite(left) & np.isfinite(right)
                self.reference_left, self.reference_right = left.copy(), right.copy()
                self._previous_left, self._previous_right = left.copy(), right.copy()
            if np.count_nonzero(valid) >= 6:
                self.reference_width = float(np.nanmedian(right - left) * self.initial_pose.length / CANONICAL_LENGTH)
                self.endpoint_tracker.reference_body_ratio = self.reference_width / self.initial_pose.length
                if not self._seen_hand:
                    self._unmasked_reference_image = pose.rectify(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype(float)
            else:
                self.reference_left = self.reference_right = None
        if self.reference_left is not None:
            known = np.isfinite(self.reference_left) & np.isfinite(self.reference_right)
            valid &= known
            left[~valid] = right[~valid] = np.nan
            # Track each observed side even when the opposite side is hidden.
            # Requiring a paired width here leaves a visibly moving contour
            # stuck at its old search position until it disappears as well.
            # These motion priors never replace the deformation baseline or
            # create a paired width from a single observed boundary.
            for previous, visible in ((self._previous_left, self.visible_left),
                                      (self._previous_right, self.visible_right)):
                observed = known & np.isfinite(visible)
                previous[observed] = visible[observed]
        metrics = self._deformation(left, right, timestamp)
        metrics['hand_occluded_profiles'] = int(np.count_nonzero(self._occluded_profiles))
        metrics['left_edge_coverage'] = float(np.mean(np.isfinite(self.visible_left)))
        metrics['right_edge_coverage'] = float(np.mean(np.isfinite(self.visible_right)))
        self.joint_estimator.update(pose, track.points, self.outline_segments + self.body_outline_segments,
                                    self.body_mask, timestamp, hand_side=track.side, angles=hand_angles)
        metrics.update(self.joint_estimator.metrics)
        edges = []
        for index in np.flatnonzero(valid):
            y = 32 + self.positions[index] * CANONICAL_LENGTH
            pair = pose.image_points([[left[index], y], [right[index], y]])
            edges.append(tuple(tuple(round(float(v)) for v in point) for point in pair))
        width = None
        if self.reference_left is not None and np.count_nonzero(valid) >= 6:
            reference = self.reference_right - self.reference_left
            width = float(self.reference_width * np.median((right - left)[valid] / reference[valid]))
        reason = "Axis + contour OK" if width is not None else "Axis OK; body edges hidden/weak"
        if width is not None and np.mean(valid) < .6:
            reason = "Partial contour - hidden dents unmeasured"
        elif width is None and np.any(self._occluded_profiles):
            reason = 'Hand excluded; show body edges for baseline' if self.baseline_left is None else 'Hand excluded; too few visible edges'
        quality = axis_quality * float(np.mean(valid))
        self.last = OutlineMeasurement(
            width_px=width, points=edges[len(edges) // 2] if width is not None else None,
            edges=tuple(edges), band_widths=self._tuple((right - left) * self.initial_pose.length / CANONICAL_LENGTH),
            quality=quality, reason=reason, axis_points=pose.points, axis_quality=axis_quality,
            profile_left=self._tuple(left), profile_right=self._tuple(right), metrics=metrics,
            joint_estimated_left=self._tuple(self.joint_estimator.profile_left),
            joint_estimated_right=self._tuple(self.joint_estimator.profile_right))
        return self.last

    def context(self):
        return {
            "bottle_method": "cap_base_aligned_contour_v1", "axis_seed_source": self.source,
            "axis_seed_is_model_estimate": self.source.startswith("learned_"),
            "initial_cap_px": self.initial_pose.cap.tolist(), "initial_base_px": self.initial_pose.base.tolist(),
            "canonical_length": CANONICAL_LENGTH, "canonical_cap_xy": CAP_POSITION.tolist(),
            "reference_width_px": self.reference_width, "width_units": "initial_axis_scale_px",
            "pose_model": "2d_similarity", "minimum_valid_profiles": 6,
            "area_minimum_coverage": .6,
            "body_occlusion_method": "landmark_seeded_sideways_local_chroma_v2",
            "hand_missing_after_detection": "body_measurement_paused",
            "outline_display": "independent_observed_sides_with_neck_and_base",
            "dense_display_samples_used_for_measurement": False,
            "unmasked_reference_revisions": self.reference_revision,
            "outline_extensions_used_for_measurement": False,
            "profile_positions": self.positions.tolist(),
            "reference_left": self._tuple(self.reference_left) if self.reference_left is not None else [],
            "reference_right": self._tuple(self.reference_right) if self.reference_right is not None else [],
            "baseline_left": self._tuple(self.baseline_left) if self.baseline_left is not None else [],
            "baseline_right": self._tuple(self.baseline_right) if self.baseline_right is not None else [],
            "baseline_noise_pct": self.baseline_noise,
            "joint_estimate": self.joint_estimator.context(),
            "joint_hand_tracking": {"method": "bidirectional_lk_with_landmark_checks",
                                    "flow_only_max_gap_s": self.joint_hand.max_gap_s,
                                    "flow_only_is_display_only": True},
        }

    def draw(self, video):
        if self.last.axis_points is None:
            return
        cap, base = (tuple(np.rint(point).astype(int)) for point in self.last.axis_points)
        cv2.line(video, cap, base, (100, 220, 240), 2, cv2.LINE_AA)
        for point, name in ((cap, "CAP"), (base, "BASE")):
            cv2.circle(video, point, 7, (100, 220, 240), 2, cv2.LINE_AA)
            cv2.putText(video, name, (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (100, 220, 240), 1, cv2.LINE_AA)
        valid = np.array([v is not None for v in self.last.profile_left])
        body = np.zeros_like(video)
        if self.baseline_left is not None:
            for side in (self.baseline_left, self.baseline_right):
                previous = None
                for index, value in enumerate(side):
                    if not np.isfinite(value) or index >= len(valid) or not valid[index]:
                        previous = None
                        continue
                    y = 32 + self.positions[index] * CANONICAL_LENGTH
                    point = tuple(np.rint(self.pose.image_points([[value, y]])[0]).astype(int))
                    if previous is not None:
                        cv2.line(body, previous, point, (150, 150, 150), 1, cv2.LINE_AA)
                    previous = point
        bottle_color = (255, 205, 70)
        for segment in self.outline_segments + self.body_outline_segments:
            cv2.polylines(body, [np.rint(segment).astype(np.int32)], False, bottle_color, 2, cv2.LINE_AA)
        for side in (self.visible_left, self.visible_right):
            previous = None
            for index, value in enumerate(side):
                if not np.isfinite(value):
                    previous = None
                    continue
                y = 32 + self.positions[index] * CANONICAL_LENGTH
                point = tuple(np.rint(self.pose.image_points([[value, y]])[0]).astype(int))
                # Every observed side gets a blue point, including isolated
                # samples. Blue no longer requires a paired width measurement.
                cv2.circle(body, point, 2, bottle_color, -1, cv2.LINE_AA)
                if previous is not None:
                    cv2.line(body, previous, point, bottle_color, 2, cv2.LINE_AA)
                previous = point
        ink = np.any(body != 0, axis=2) & (self.body_mask == 0)
        video[ink] = body[ink]
        self.joint_estimator.draw(video)
