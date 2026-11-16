"""Measure visible bottle edges independently of the detector's bounding box."""

from dataclasses import dataclass, field

import cv2
import numpy as np


Point = tuple[int, int]
Rectangle = tuple[int, int, int, int]
EdgePair = tuple[Point, Point]


def hand_mask(shape: tuple[int, ...], hand: list[Point]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(hand) != 21:
        return mask
    points = np.asarray(hand, dtype=np.int32)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    radius = max(5, round(palm_width * 0.12))
    # A hull around ALL landmarks also erases visible gaps between fingers.
    palm = cv2.convexHull(points[[0, 1, 2, 5, 9, 13, 17]])
    cv2.fillConvexPoly(mask, palm, 255)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20)):
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, tuple(points[a]), tuple(points[b]), 255, 2 * radius)
        for index in chain:
            cv2.circle(mask, tuple(points[index]), radius, 255, -1)
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))


def edge_image(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Local contrast helps faint boundaries, but cannot recover invisible ones.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vertical = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    horizontal[vertical > np.abs(horizontal) * 1.6] = 0
    return horizontal


@dataclass(frozen=True)
class Edge:
    x: float
    slope: float
    support: float
    polarity: int


def find_edge(strength: np.ndarray, blocked: np.ndarray, expected: float,
              radius: float, polarity: int = 0) -> Edge | None:
    height, width = strength.shape
    start, end = max(1, int(expected - radius)), min(width - 1, int(expected + radius + 1))
    if end - start < 3 or height < 6:
        return None
    window = (np.maximum(0, strength[:, start:end] * polarity) if polarity
              else np.abs(strength[:, start:end]))
    window[blocked[:, start:end] > 0] = 0
    columns = np.arange(start, end)
    prior = np.exp(-0.5 * ((columns - expected) / max(3, radius * 0.48)) ** 2)
    peaks = np.argmax(window * prior, axis=1)
    rows = np.arange(height)
    values = window[rows, peaks]
    positions = columns[peaks].astype(float)
    valid = values >= 12.0
    required = max(6, int(np.ceil(height * 0.40)))
    if np.count_nonzero(valid) < required:
        return None
    yy = rows - (height - 1) / 2
    center = np.median(positions[valid])
    valid &= np.abs(positions - center) <= max(4, radius * 0.45)
    for _ in range(3):
        if np.count_nonzero(valid) < required:
            return None
        slope, intercept = np.polyfit(yy[valid], positions[valid], 1)
        valid &= np.abs(positions - (slope * yy + intercept)) <= 2.5
    if np.count_nonzero(valid) < required or abs(slope) > 0.7:
        return None
    slope, intercept = np.polyfit(yy[valid], positions[valid], 1)
    signs = strength[rows[valid], positions[valid].astype(int)]
    sign = 1 if np.median(signs) >= 0 else -1
    return Edge(float(intercept), float(slope), float(np.mean(valid)), sign)


@dataclass
class Profile:
    y: int
    height: int
    reference_left: float
    reference_right: float
    left: float
    right: float
    left_polarity: int
    right_polarity: int

    @property
    def reference_width(self) -> float:
        return self.reference_right - self.reference_left


@dataclass(frozen=True)
class OutlineMeasurement:
    width_px: float | None = None
    points: EdgePair | None = None
    edges: tuple[EdgePair, ...] = ()
    band_widths: tuple[float | None, ...] = ()
    quality: float = 0.0
    reason: str = "Select the bottle body with E"
    axis_points: tuple[tuple[float, float], tuple[float, float]] | None = None
    axis_quality: float = 0.0
    profile_left: tuple[float | None, ...] = ()
    profile_right: tuple[float | None, ...] = ()
    metrics: dict[str, float | None] = field(default_factory=dict)
    joint_estimated_left: tuple[float | None, ...] = ()
    joint_estimated_right: tuple[float | None, ...] = ()


@dataclass
class BottleBand:
    region: Rectangle
    profiles: list[Profile]
    reference_width: float
    last: OutlineMeasurement = field(default_factory=OutlineMeasurement)
    offset_x: float = 0.0

    @classmethod
    def select(cls, frame: np.ndarray, region: Rectangle,
               hand: list[Point]) -> "BottleBand | None":
        x, y, width, height = (int(value) for value in region)
        if width < 45 or height < 18:
            return None
        if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
            return None
        strength, blocked = edge_image(frame), hand_mask(frame.shape, hand)
        count = min(5, max(1, height // 25))
        band_height = min(25, max(18, height // count))
        centers = np.linspace(y + band_height / 2, y + height - band_height / 2, count)
        profiles = []
        for center in centers:
            top = round(center - band_height / 2)
            patch = strength[top:top + band_height, x:x + width]
            mask = blocked[top:top + band_height, x:x + width]
            left = find_edge(patch, mask, width * 0.12, width * 0.19)
            right = find_edge(patch, mask, width * 0.88, width * 0.19)
            if left is None or right is None or right.x - left.x < max(20, width * 0.40):
                continue
            if abs(left.slope - right.slope) > 0.45:
                continue
            # Lock gradient polarity before storing the reference. Otherwise a
            # thin transparent rim can alternate between its two gradient lobes.
            radius = max(9, (right.x - left.x) * 0.16)
            left = find_edge(patch, mask, left.x, radius, left.polarity)
            right = find_edge(patch, mask, right.x, radius, right.polarity)
            if left is None or right is None:
                continue
            if right.x - left.x < max(20, width * 0.40) or abs(left.slope - right.slope) > 0.45:
                continue
            profiles.append(Profile(top, band_height, x + left.x, x + right.x,
                                    x + left.x, x + right.x, left.polarity, right.polarity))
        if not profiles:
            return None
        reference = float(np.median([profile.reference_width for profile in profiles]))
        result = cls((x, y, width, height), profiles, reference)
        result.measure(frame, hand)
        return result

    def measure(self, frame: np.ndarray, hand: list[Point]) -> OutlineMeasurement:
        strength, blocked = edge_image(frame), hand_mask(frame.shape, hand)
        ratios, edges, supports, centers = [], [], [], []
        band_widths: list[float | None] = []
        occluded = 0
        for profile in self.profiles:
            top, height = profile.y, profile.height
            band_widths.append(None)
            if top < 0 or top + height > frame.shape[0]:
                continue
            radius = max(9, profile.reference_width * 0.16)
            patch, mask = strength[top:top + height], blocked[top:top + height]
            # Do not substitute a finger's boundary for an occluded bottle edge.
            for position in (profile.left, profile.right):
                lo, hi = max(0, round(position - 4)), min(frame.shape[1], round(position + 5))
                if hi > lo and np.mean(mask[:, lo:hi] > 0) > 0.6:
                    occluded += 1
                    break
            else:
                left = find_edge(patch, mask, profile.left, radius, profile.left_polarity)
                right = find_edge(patch, mask, profile.right, radius, profile.right_polarity)
                if left is None or right is None:
                    continue
                width = right.x - left.x
                if not 0.45 * profile.reference_width <= width <= 1.5 * profile.reference_width:
                    continue
                if abs(left.slope - right.slope) > 0.45:
                    continue
                profile.left, profile.right = left.x, right.x
                mid_y = top + (height - 1) // 2
                edges.append(((round(left.x), mid_y), (round(right.x), mid_y)))
                ratios.append(width / profile.reference_width)
                band_widths[-1] = float(width)
                supports.append(min(left.support, right.support))
                centers.append((left.x + right.x - profile.reference_left - profile.reference_right) / 2)
        minimum = min(2, len(self.profiles))
        if len(edges) < minimum:
            reason = "Hand covers bottle edges" if occluded else "Edges weak / bottle moved - E to reselect"
            self.last = OutlineMeasurement(band_widths=tuple(band_widths), reason=reason)
            return self.last
        self.offset_x = float(np.median(centers))
        width = self.reference_width * float(np.median(ratios))
        quality = float(np.mean(supports) * len(edges) / len(self.profiles))
        # Line endpoints belong to one real band; width_px combines all bands.
        self.last = OutlineMeasurement(width, edges[len(edges) // 2], tuple(edges),
                                       tuple(band_widths), quality, "Visible edges OK")
        return self.last

    @property
    def display_region(self) -> Rectangle:
        x, y, width, height = self.region
        return round(x + self.offset_x), y, width, height
