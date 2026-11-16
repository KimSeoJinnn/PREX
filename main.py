from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np

from bottle_detector import BottleDetector, model_file as bottle_model_file
from bottle_axis import AXIS_METRICS, AxisBottleTracker
from bottle_auto import AutomaticBottleFinder
from bottle_learned import LearnedBottleFinder, AsyncBottleFinder, BottleSearchProcessor
from bottle_outline import hand_mask
from bottle_outline import BottleBand, OutlineMeasurement
from grip_features import FRAME_NAMES, frame_features
from grip_runtime import GripExperiment, load_settings, label_metadata


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

JOINTS = {
    "Thumb MCP": (1, 2, 3),
    "Thumb IP": (2, 3, 4),
    "Index MCP": (0, 5, 6),
    "Index PIP": (5, 6, 7),
    "Index DIP": (6, 7, 8),
    "Middle MCP": (0, 9, 10),
    "Middle PIP": (9, 10, 11),
    "Middle DIP": (10, 11, 12),
    "Ring MCP": (0, 13, 14),
    "Ring PIP": (13, 14, 15),
    "Ring DIP": (14, 15, 16),
    "Pinky MCP": (0, 17, 18),
    "Pinky PIP": (17, 18, 19),
    "Pinky DIP": (18, 19, 20),
}
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
MARKER_NAMES = ("blue", "pink")
# OpenCV hue runs from 0 to 179. Use matte, saturated stickers.
MARKER_HSV = (((93, 115, 70), (132, 255, 255)), ((140, 115, 70), (175, 255, 255)))
MARKER_COLORS = ((255, 172, 74), (205, 106, 255))
KERNEL = np.ones((3, 3), dtype=np.uint8)

WINDOW = "PREX | Webcam grip lab"
WIDTH, HEIGHT = 640, 480
TREND_SECONDS = 8.0
BASELINE_SECONDS = 1.0
MIN_BASELINE_SAMPLES = 3
MIN_BASELINE_DISTANCE = 20.0

INK = (235, 240, 245)
MUTED = (163, 175, 190)
LIME = (153, 224, 110)
AQUA = (223, 197, 88)
PINK = (205, 106, 255)
WARN = (91, 166, 255)

Point = tuple[int, int]


@dataclass
class Hand:
    side: str
    points: list[Point]
    angles: np.ndarray | None


@dataclass
class FrameData:
    hand: Hand | None
    bottle_points: tuple[Point | None, Point | None]
    outline: OutlineMeasurement | None = None

    @property
    def distance(self) -> float | None:
        if self.outline is not None:
            return self.outline.width_px
        first, second = self.bottle_points
        if first is None or second is None:
            return None
        return math.dist(first, second)


@dataclass
class Baseline:
    hand_side: str | None = None
    angles: np.ndarray | None = None
    distance: float | None = None

    def changes(self, frame: FrameData) -> tuple[np.ndarray | None, float | None]:
        deltas = None
        if self.angles is not None and frame.hand and frame.hand.side == self.hand_side:
            if frame.hand.angles is not None:
                deltas = frame.hand.angles - self.angles
        distance = frame.distance
        compression = None
        if self.distance is not None and distance is not None:
            compression = 100.0 * (self.distance - distance) / self.distance
        if frame.outline and "median_width_reduction_pct" in frame.outline.metrics:
            compression = frame.outline.metrics["median_width_reduction_pct"]
        return deltas, compression


class Calibration:
    def __init__(self, starts_at: float):
        self.starts_at = starts_at
        self.ends_at = starts_at + BASELINE_SECONDS
        self.hands: dict[str, list[np.ndarray]] = defaultdict(list)
        self.distances: list[float] = []
        self.outlines: list[OutlineMeasurement] = []

    def add(self, frame: FrameData) -> None:
        if frame.hand and frame.hand.angles is not None:
            self.hands[frame.hand.side].append(frame.hand.angles)
        distance = frame.distance
        if distance is not None and distance >= MIN_BASELINE_DISTANCE:
            self.distances.append(distance)
            if frame.outline and frame.outline.profile_left:
                self.outlines.append(frame.outline)

    def finish(self) -> Baseline:
        baseline = Baseline()
        if self.hands:
            side, samples = max(self.hands.items(), key=lambda pair: len(pair[1]))
            if len(samples) >= MIN_BASELINE_SAMPLES:
                baseline.hand_side = side
                baseline.angles = np.median(samples, axis=0)
        if len(self.distances) >= MIN_BASELINE_SAMPLES:
            baseline.distance = float(np.median(self.distances))
        return baseline


def model_file() -> Path:
    if MODEL_PATH.is_file():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the Google MediaPipe Hand Landmarker model...", flush=True)
    temp_path = None
    try:
        with NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as target:
            temp_path = Path(target.name)
            with urlopen(MODEL_URL, timeout=30) as source:
                shutil.copyfileobj(source, target)
        with temp_path.open("rb") as downloaded:
            if hashlib.file_digest(downloaded, "sha256").hexdigest() != MODEL_SHA256:
                raise OSError("Model checksum mismatch.")
        temp_path.replace(MODEL_PATH)
    except OSError as error:
        raise RuntimeError(f"Model download failed. See README.md. ({error})") from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return MODEL_PATH


def joint_angles(points: np.ndarray) -> np.ndarray | None:
    # MCP angles on the four long fingers use the wrist as the palm-side proxy.
    triples = np.asarray(list(JOINTS.values()))
    first, center, last = points[triples].transpose(1, 0, 2)
    before, after = first - center, last - center
    magnitudes = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    if not np.isfinite(points).all() or np.any(magnitudes < 1e-10):
        return None
    cosines = np.sum(before * after, axis=1) / magnitudes
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def marker_center(hsv: np.ndarray, limits: tuple[tuple[int, int, int], tuple[int, int, int]]) -> Point | None:
    mask = cv2.inRange(hsv, limits[0], limits[1])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    max_area = hsv.shape[0] * hsv.shape[1] * 0.012
    for contour in contours:
        area = cv2.contourArea(contour)
        if 25 <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            if 4 * math.pi * area / (perimeter * perimeter) >= 0.45:
                candidates.append((area, contour))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[1][0] > 0.6 * candidates[0][0]:
        return None  # Multiple plausible stickers for this color.
    moments = cv2.moments(candidates[0][1])
    return (
        round(moments["m10"] / moments["m00"]),
        round(moments["m01"] / moments["m00"]),
    )


class LiveHandModels:
    """Separate tracking state for full-image and bottle-neighborhood views."""

    def __init__(self):
        def create(confidence):
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file())),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
                min_tracking_confidence=.3)
            return mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.full_model = create(.35)
        try:
            self.roi_model = create(.2)
        except Exception:
            self.full_model.close()
            raise
        self.prefer_roi = False
        self.roi_view = None

    def detect_for_video(self, image, timestamp_ms):
        return self.full_model.detect_for_video(image, timestamp_ms)

    def close(self):
        try:
            self.full_model.close()
        finally:
            self.roi_model.close()


def observe_hand(frame, landmarker, timestamp_ms, region=None):
    """Try the full image, then a bottle neighborhood for a partly hidden hand.

    Each attempt uses a unique timestamp. A missed frame produces no angles;
    we never fill a measurement row with a previous hand result.
    """
    h, w = frame.shape[:2]
    views = [(0, 0, w, h, False)]
    if region is not None:
        x, y, bw, bh = region
        # A square neighborhood keeps the palm detector's input geometry
        # stable as the bottle changes aspect ratio.
        side = min(min(w, h), round(max(bh * 1.14, bw * 2.5)))
        x0 = int(np.clip(round(x + bw / 2 - side / 2), 0, w - side))
        y0 = int(np.clip(round(y + bh / 2 - side / 2), 0, h - side))
        if side >= 80 and side * side < w * h * .9:
            view = (x0, y0, x0 + side, y0 + side, True)
            previous = getattr(landmarker, 'roi_view', None)
            if previous is not None:
                old_side = previous[2] - previous[0]
                old_center = np.array([(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2])
                center = np.array([x0 + side / 2, y0 + side / 2])
                if (.85 <= side / old_side <= 1.15 and np.linalg.norm(center - old_center) <= old_side * .12
                        and previous[2] <= w and previous[3] <= h):
                    view = previous
            # Switching from a DNN box to a tracked-axis box should not resize
            # the hand model's crop on an otherwise stationary bottle. This
            # retains only crop geometry; every landmark is still inferred anew.
            if hasattr(landmarker, 'roi_view'):
                landmarker.roi_view = view
            views.append(view)
    if getattr(landmarker, 'prefer_roi', False):
        views.reverse()
    for attempt, (x0, y0, x1, y1, is_roi) in enumerate(views):
        rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        model = getattr(landmarker, 'roi_model', landmarker) if is_roi else landmarker
        result = model.detect_for_video(image, timestamp_ms * 3 + attempt)
        if not result.hand_landmarks:
            continue
        points = [(round(p.x * (x1 - x0) + x0), round(p.y * (y1 - y0) + y0)) for p in result.hand_landmarks[0]]
        side = result.handedness[0][0].category_name
        world = np.asarray([(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]])
        if hasattr(landmarker, 'prefer_roi'):
            landmarker.prefer_roi = is_roi
        return Hand(side, points, joint_angles(world))
    return None


def observe(
    frame: np.ndarray, landmarker, timestamp_ms: int,
    markerless: bool = False, band: BottleBand | AxisBottleTracker | None = None,
    hand_region=None,
) -> FrameData:
    hand = observe_hand(frame, landmarker, timestamp_ms, hand_region)
    if markerless:
        if isinstance(band, AxisBottleTracker):
            outline = band.measure(frame, hand.points if hand else [], timestamp_ms / 1000,
                                   hand_side=hand.side if hand else None, hand_angles=hand.angles if hand else None)
        else:
            outline = band.measure(frame, hand.points if hand else []) if band else OutlineMeasurement()
        return FrameData(hand, outline.points or (None, None), outline)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, pink = (marker_center(hsv, limits) for limits in MARKER_HSV)
    return FrameData(hand, (blue, pink))


def select_bottle_band(frame: np.ndarray, hand: Hand | None) -> BottleBand | None:
    name = "Bottle BODY + small side margins | drag, Enter; Esc cancels"
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    try:
        region = cv2.selectROI(name, frame, showCrosshair=False)
    finally:
        cv2.destroyWindow(name)
    return BottleBand.select(frame, region, hand.points if hand else [])


def select_bottle_axis(frame, hand, width_hint=None):
    name = "Click CAP center, then BASE center | Enter accept | right-click undo | Esc cancel"
    points = []

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, click)
    try:
        while True:
            canvas = frame.copy()
            label(canvas, "1 CAP center   2 BASE center   Enter", 10, 24, .55, AQUA)
            for index, point in enumerate(points):
                cv2.circle(canvas, point, 7, AQUA, 2, cv2.LINE_AA)
                label(canvas, "CAP" if index == 0 else "BASE", point[0] + 10, point[1], .5, AQUA)
            if len(points) == 2:
                cv2.line(canvas, points[0], points[1], AQUA, 2, cv2.LINE_AA)
            cv2.imshow(name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key in (8, 127) and points:
                points.pop()
            if key in (10, 13, 32) and len(points) == 2:
                try:
                    return AxisBottleTracker(frame, *points, hand.points if hand else [], width_hint)
                except ValueError as error:
                    print(f"Axis selection: {error}", file=sys.stderr)
                    points.clear()
    finally:
        cv2.destroyWindow(name)


def axis_columns(outline):
    points = outline.axis_points or ((None, None), (None, None))
    return [
        outline.axis_quality if outline.axis_points else "",
        *[value if value is not None else "" for point in points for value in point],
        json.dumps(outline.profile_left), json.dumps(outline.profile_right),
        *["" if outline.metrics.get(name) is None else round(outline.metrics[name], 5)
          for name in AXIS_METRICS],
        json.dumps(outline.joint_estimated_left), json.dumps(outline.joint_estimated_right),
    ]


def format_number(value: float | None, digits=1, signed=False) -> str:
    if value is None:
        return "--"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def write_row(writer, seconds, frame, baseline, deltas, compression, metadata=None):
    hand = frame.hand
    angles = hand.angles if hand else None
    writer.writerow([
        round(seconds, 3),
        hand.side if hand else "",
        int(frame.distance is not None),
        "" if frame.distance is None else round(frame.distance, 3),
        "" if baseline.distance is None else round(baseline.distance, 3),
        "" if compression is None else round(compression, 3),
        *[coordinate for point in frame.bottle_points for coordinate in (point or ("", ""))],
        *([round(float(value), 3) for value in angles] if angles is not None else [""] * len(JOINTS)),
        *([round(float(value), 3) for value in deltas] if deltas is not None else [""] * len(JOINTS)),
        *([
            round(frame.outline.quality, 3), frame.outline.reason,
            sum(value is not None for value in frame.outline.band_widths),
            json.dumps(frame.outline.band_widths),
            *axis_columns(frame.outline),
        ] if frame.outline is not None else []),
        *(metadata or {}).values(),
    ])


def open_recording(output_dir: Path, markerless: bool = False, metadata=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "outline" if markerless else "trial"
    path = output_dir / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.csv")
    stream = path.open("x", newline="", encoding="utf-8")
    try:
        writer = csv.writer(stream)
        names = [name.lower().replace(" ", "_") for name in JOINTS]
        if markerless:
            bottle_fields = [
                "outline_visible", "bottle_width_px", "bottle_baseline_width_px",
                "bottle_width_reduction_pct",
            ]
            point_names = ("outline_left", "outline_right")
        else:
            bottle_fields = [
                "markers_visible", "bottle_distance_px", "bottle_baseline_px",
                "bottle_compression_pct",
            ]
            point_names = MARKER_NAMES
        writer.writerow([
            "time_s", "hand", *bottle_fields,
            *[f"{name}_{axis}_px" for name in point_names for axis in ("x", "y")],
            *[f"{name}_deg" for name in names],
            *[f"{name}_delta_deg" for name in names],
            *([
                "outline_quality", "outline_status", "visible_bands", "band_widths_px",
                "axis_quality", "cap_x_px", "cap_y_px", "base_x_px", "base_y_px",
                "profile_left_canonical_px", "profile_right_canonical_px", *AXIS_METRICS,
                "joint_estimated_left_canonical_px", "joint_estimated_right_canonical_px",
            ] if markerless else []),
            *(metadata or {}).keys(),
        ])
    except Exception:
        stream.close()
        raise
    print(f"CSV: {path}", flush=True)
    return stream, writer


def label(canvas, value, x, y, size=0.52, color=INK, thickness=1):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def chart(canvas, history, now, index, top, title, unit, color, min_limit):
    left, right, height = 14, 625, 37
    samples = [(sample[0], sample[index]) for sample in history]
    limit = max([min_limit, *(abs(value) for _, value in samples if value is not None)])
    cv2.rectangle(canvas, (left, top), (right, top + height), (38, 47, 57), -1)
    cv2.line(canvas, (left, top + height // 2), (right, top + height // 2), (73, 86, 100), 1)
    last = None
    for seconds, value in samples:
        if value is None:
            last = None
            continue
        point = (
            round(right - (now - seconds) * (right - left) / TREND_SECONDS),
            round(top + height / 2 - value * (height / 2 - 2) / limit),
        )
        if last is not None:
            cv2.line(canvas, last, point, color, 2, cv2.LINE_AA)
        last = point
    label(canvas, title, left, top - 6, 0.45, color)
    label(canvas, f"{format_number(samples[-1][1], signed=True)} {unit}",
          right - 100, top - 6, 0.45, color)


def preview(
    frame, data, deltas, compression, baseline, history, now, status, recording,
    markerless=False, band=None, detection=None, automatic=False,
):
    canvas = np.full((650, 1000, 3), (25, 31, 40), dtype=np.uint8)
    height, width = frame.shape[:2]
    video = canvas[:height, :width]
    video[:] = frame
    if automatic and band is None:
        cv2.rectangle(video, (round(width * .20), 35), (round(width * .80), height - 30), MUTED, 1)
        label(video, "Show whole bottle, keep grip relaxed", 10, 24, .49, AQUA)

    if detection is not None and band is None:
        bx, by, bw, bh = detection.region
        cv2.rectangle(video, (bx, by), (bx + bw, by + bh), AQUA, 2)
        label(video, f"Bottle {detection.score:.0%} (location only)", bx, max(18, by - 6), 0.48, AQUA)
        if getattr(detection, "mask", None) is not None:
            contours, _ = cv2.findContours(detection.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(video, contours, -1, AQUA, 1)

    if data.hand:
        points = data.hand.points
        for start, end in HAND_EDGES:
            cv2.line(video, points[start], points[end], LIME, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(video, point, 3, INK, -1, cv2.LINE_AA)

    first, second = data.bottle_points
    if isinstance(band, AxisBottleTracker):
        band.draw(video)
        if band.last.axis_points is not None:
            label(video, "BLUE: bottle edges | GREEN: hand", 10, 50, .47, AQUA)
            if band.joint_estimator.segments:
                text = ("PINK: tracking only; readings paused" if band.joint_estimator.display_only
                        else "PINK DASHES: joint ESTIMATE")
                label(video, text, 10, 70, .47, PINK)
    elif band:
        band_x, band_y, band_w, band_h = band.display_region
        cv2.rectangle(video, (band_x, band_y), (band_x + band_w, band_y + band_h), MUTED, 1)
        for index, profile in enumerate(band.profiles):
            band_color = AQUA if band.last.band_widths[index] is not None else WARN
            cv2.line(video, (max(0, band_x), profile.y),
                     (min(width - 1, band_x + band_w), profile.y), band_color, 1)
        for left, right in band.last.edges:
            cv2.line(video, left, right, AQUA, 2, cv2.LINE_AA)
    if not markerless and first is not None and second is not None:
        cv2.line(video, first, second, AQUA, 2, cv2.LINE_AA)
    colors = (AQUA, AQUA) if markerless else MARKER_COLORS
    for point, color in zip(data.bottle_points, colors):
        if point is not None:
            cv2.circle(video, point, 5 if markerless else 8, color, 2, cv2.LINE_AA)

    x = WIDTH + 18
    cv2.line(canvas, (WIDTH, 0), (WIDTH, 649), (61, 73, 89), 1)
    label(canvas, "HAND & BOTTLE", x, 31, 0.76, INK, 2)
    label(canvas, f"Hand: {data.hand.side if data.hand else 'not detected'}", x, 58, 0.51, LIME)
    if markerless:
        located = data.outline is not None and data.outline.axis_points is not None
        bottle_state = 'tracked' if located else ('recovering' if band is not None else 'searching')
        label(canvas, f'Bottle: {bottle_state}', x + 169, 58, .41, LIME if located else WARN)
    label(canvas, "JOINT", x, 89, 0.49, MUTED)
    label(canvas, "ANGLE", x + 143, 89, 0.49, MUTED)
    label(canvas, "DELTA", x + 238, 89, 0.49, MUTED)
    angles = data.hand.angles if data.hand else None
    for index, name in enumerate(JOINTS):
        y = 112 + index * 23
        label(canvas, name, x, y, 0.49)
        label(canvas, format_number(angles[index] if angles is not None else None),
              x + 143, y, 0.49)
        label(canvas, format_number(deltas[index] if deltas is not None else None, signed=True),
              x + 238, y, 0.49, LIME)

    cv2.line(canvas, (x, 426), (981, 426), (61, 73, 89), 1)
    quantity = "Effective width" if markerless else "Bottle distance"
    trend = "Width reduction" if markerless else "Bottle compression"
    reading = "Width reduction" if markerless else "Compression"
    controls = "A Restart B Base R Rec D Save Q Quit" if automatic else (
        "P 2pts  A Auto  B Base  R Rec  Q Quit" if markerless else "B  Baseline     R  Record     Q  Quit")
    if markerless:
        visible = sum(value is not None for value in data.outline.band_widths) if data.outline else 0
        total = (band.profile_count if isinstance(band, AxisBottleTracker) else len(band.profiles)) if band else 0
        quality = data.outline.quality if data.outline else 0
        bottle_status = f"Visible bands {visible}/{total} | Quality {quality:.0%}"
        hidden = data.outline.metrics.get('hand_occluded_profiles') if data.outline else None
        if hidden is not None:
            bottle_status = f'Visible {visible}/{total} | Hand hides {int(hidden)}'
        if isinstance(band, AxisBottleTracker):
            left_visible = int(np.count_nonzero(np.isfinite(band.visible_left)))
            right_visible = int(np.count_nonzero(np.isfinite(band.visible_right)))
            bottle_status = f'Edges L {left_visible} R {right_visible} | Width {visible}/{total}'
    else:
        bottle_status = "Blue  " + ("OK" if first is not None else "--")
        bottle_status += "    Pink  " + ("OK" if second is not None else "--")
    if isinstance(band, AxisBottleTracker):
        estimated = data.outline.metrics.get('joint_estimated_width_reduction_pct') if data.outline else None
        label(canvas, f"Visible width  {format_number(data.distance)} px", x, 451, .49, AQUA)
        label(canvas, f"Visible change  {format_number(compression, signed=True)} %", x, 474, .49, AQUA)
        label(canvas, f"Joint EST  {format_number(estimated, signed=True)} %", x, 500, .57, PINK, 2)
        label(canvas, band.joint_estimator.status, x, 524, .40, PINK)
        controls = 'A Reset H Base G Trial R Rec D Save Q Quit' if automatic else 'P Axis B Base H Est-base R Rec Q Quit'
    else:
        label(canvas, f"{quantity}   {format_number(data.distance)} px", x, 451, 0.51, AQUA)
        label(canvas, f"Bottle baseline   {format_number(baseline.distance)} px", x, 474, 0.51, MUTED)
        label(canvas, f"{reading}    {format_number(compression, signed=True)} %",
              x, 500, 0.60, AQUA, 2)
        label(canvas, bottle_status, x, 524, 0.45, MUTED)
    if markerless and data.outline:
        label(video, data.outline.reason, 10, height - 12, 0.48,
              AQUA if data.distance is not None else WARN)
        if data.outline.axis_points:
            metrics = data.outline.metrics
            label(video, f"Dent P95: {format_number(metrics.get('dent_p95_pct'))}%", 10, height - 64, .49, AQUA)
            label(video, f"Visible area: {format_number(metrics.get('visible_area_reduction_pct'))}%"
                  f" | coverage {format_number(100 * metrics.get('profile_coverage', 0))}%",
                  10, height - 39, .49, AQUA)
    label(canvas, status, x, 552, 0.48, WARN)
    label(canvas, "REC  Saving CSV" if recording else "IDLE", x, 576, 0.51,
          PINK if recording else MUTED, 2)
    label(canvas, controls, x, 605, 0.43 if isinstance(band, AxisBottleTracker) else 0.51, INK)
    label(canvas, "Angles: deg  |  Trends: last 8 s", x, 627, 0.47, MUTED)

    chart(canvas, history, now, 1, 512, "Mean absolute joint-angle change", "deg", LIME, 10)
    chart(canvas, history, now, 2, 584, trend, "%", AQUA, 5)
    return canvas


def baseline_status(baseline: Baseline) -> str:
    hand = "OK" if baseline.angles is not None else "--"
    bottle = "OK" if baseline.distance is not None else "--"
    return f"Baseline  Hand {hand}  |  Bottle {bottle}"


def save_recording_context(stream, band, baseline):
    context = band.context() if isinstance(band, AxisBottleTracker) else {
        "bottle_method": "reference_normalized_visible_bands" if band else "markers_or_no_outline",
        "region_xywh": band.region if band else None,
        "reference_width_px": band.reference_width if band else None,
        "profiles": [vars(profile) for profile in band.profiles] if band else [],
    }
    context.update({
        "schema_version": 3,
        "baseline_width_px": baseline.distance,
        "baseline_hand": baseline.hand_side,
        "baseline_angles_deg": baseline.angles.tolist() if baseline.angles is not None else None,
    })
    Path(stream.name).with_suffix(".json").write_text(json.dumps(context, indent=2), encoding="utf-8")


class RealtimeSession:
    """Hand/contour measurements run independently of slow bottle inference."""

    def __init__(self, landmarker, detector, output_dir, metadata, model_name, shape_only=False, search_factory=None,
                 grip_settings=None, grip_model_path=None, grip_root=None):
        self.landmarker = landmarker
        self.finder = AsyncBottleFinder(search_factory) if search_factory else (
            AutomaticBottleFinder() if shape_only else LearnedBottleFinder(detector))
        self.output_dir, self.user_metadata = output_dir, metadata
        self.model_name = model_name
        self.band = None
        self.baseline = Baseline()
        self.calibration = None
        self.history = deque()
        self.stream = self.writer = None
        self.started_at = self.record_started_at = None
        self.retry_after = 0.
        self.last_timestamp = -1
        self._lost_since = None
        self.joint_request_pending = False
        self.grip = GripExperiment(metadata, grip_settings, root=grip_root, model_path=grip_model_path)
        self.current_features = dict.fromkeys(FRAME_NAMES)
        self.status = "Show one bottle with your grip relaxed"

    def _metadata(self, packet, now, latency):
        band = self.band if isinstance(self.band, AxisBottleTracker) else None
        return {**self.user_metadata, "source_frame_id": packet.frame_id,
                "capture_time_s": round(now, 5), "analysis_latency_ms": round(latency, 2),
                "bottle_model": self.model_name,
                "joint_hand_source": band.joint_hand.last.source if band else 'missing',
                "joint_hand_age_ms": round(band.joint_hand.last.age_ms, 2) if band else None,
                "joint_display_only": band.joint_estimator.display_only if band else False,
                "trial_phase": self.grip.phase(now),
                "trial_kind": 'maximal' if self.grip.guided else self.user_metadata.get('trial_kind', 'practice'),
                **{'feat_' + n: self.current_features.get(n) for n in FRAME_NAMES}}

    def _stop_recording(self, completed=False):
        if self.stream is not None:
            path = Path(self.stream.name)
            try:
                save_recording_context(self.stream, self.band, self.baseline)
                context_path = path.with_suffix('.json')
                context = json.loads(context_path.read_text(encoding='utf-8'))
                context.update(grip_trial_guided=self.grip.guided, grip_trial_completed=bool(completed))
                context_path.write_text(json.dumps(context, indent=2), encoding='utf-8')
            finally:
                self.stream.close()
                self.stream = self.writer = None
            self.grip.finish(path, completed)

    def process(self, packet, commands):
        started = time.perf_counter()
        if self.started_at is None:
            self.started_at = packet.captured_at
        now = packet.captured_at - self.started_at
        frame = packet.frame
        for command in commands:
            if command == 'h':
                if isinstance(self.band, AxisBottleTracker) and self.writer is None:
                    self.grip.reset_reference()
                    self.band.reset_baseline()
                    self.baseline = Baseline()
                    self.calibration = None
                    self.band.joint_estimator.request(now)
                    self.joint_request_pending = False
                elif self.writer is not None:
                    self.status = 'Stop recording before resetting estimate'
                else:
                    self.grip.reset_reference()
                    self.joint_request_pending = True
                    self.status = 'Estimate requested; waiting for bottle cap/base'
            if command == 'g' and self.writer is None:
                ready = (isinstance(self.band, AxisBottleTracker)
                         and (self.band.joint_estimator.ready or self.baseline.distance is not None)
                         and not self.band.joint_estimator.calibrating)
                if ready and not self.grip.reference_consumed:
                    self.calibration = None
                    self.grip.begin(now, guided=True)
                    self.stream, self.writer = open_recording(self.output_dir, True, self._metadata(packet, now, 0))
                    self.record_started_at = now
                    save_recording_context(self.stream, self.band, self.baseline)
                else:
                    self.grip.message = 'Relax and press H before the next guided trial'
                    self.grip.last_result = None
            if command == 'r':
                if self.writer is not None:
                    self._stop_recording()
                elif (((self.calibration is None and self.baseline.distance is not None)
                       or (isinstance(self.band, AxisBottleTracker) and self.band.joint_estimator.ready))
                      and not (isinstance(self.band, AxisBottleTracker) and self.band.joint_estimator.calibrating)):
                    self.calibration = None
                    self.grip.begin(now, guided=False)
                    self.stream, self.writer = open_recording(self.output_dir, True, self._metadata(packet, now, 0))
                    self.record_started_at = now
                    save_recording_context(self.stream, self.band, self.baseline)
                else:
                    self.status = "Need baseline: relax BODY grip, press H"
            if command in ('a', 'b') and self.writer is None:
                self.grip.reset_reference()
                self.joint_request_pending = False
                self.baseline = Baseline()
                self.calibration = None
                self.history.clear()
                self.retry_after = now
                if command == 'a':
                    self.band = None
                    self._lost_since = None
                    self.finder.reset()
                elif self.band is not None:
                    self.band.reset_baseline()
                    self.calibration = Calibration(now + 3.)
                    self.calibration.ends_at = self.calibration.starts_at + 1.5
        timestamp = max(self.last_timestamp + 1, round(now * 1000))
        self.last_timestamp = timestamp
        detection = getattr(self.finder, 'detection', None)
        hand_region = None
        if detection is not None and now - getattr(self.finder, 'detection_time', -float('inf')) < 3.:
            hand_region = detection.region
        elif self.band is not None:
            # Approximate the full bottle box, not the wide contour search ROI.
            cap, base = self.band.pose.cap, self.band.pose.base
            body_width = max(40, self.band.reference_width or np.linalg.norm(base - cap) * .33)
            hand_region = (min(cap[0], base[0]) - body_width / 2, min(cap[1], base[1]),
                           abs(base[0] - cap[0]) + body_width, abs(base[1] - cap[1]))
        revision = self.band.reference_revision if self.band is not None else None
        joint_was_calibrating = isinstance(self.band, AxisBottleTracker) and self.band.joint_estimator.calibrating
        data = observe(frame, self.landmarker, timestamp, True, self.band, hand_region)
        if self.band is not None and revision != self.band.reference_revision:
            self.baseline.distance = None
            self.calibration = None
            self.retry_after = now + .4
            if joint_was_calibrating:
                self.band.joint_estimator.request(now)
        hand = data.hand.points if data.hand else []
        if self.band is None:
            self._lost_since = None
            self.band = self.finder.acquire(frame, hand, now)
            self.status = self.finder.status
            if self.band is not None:
                data.outline = self.band.last
                data.bottle_points = data.outline.points or (None, None)
            else:
                data.outline = OutlineMeasurement(reason='AI locating bottle - keep it in view')
        elif data.outline.axis_points is None:
            if self._lost_since is None:
                self._lost_since = now
            candidate = self.finder.update(frame, hand, now, (self.band.pose.cap, self.band.pose.base))
            self.status = "Tracking lost - AI searching same bottle"
            if candidate is not None:
                if self.band.endpoint_tracker.adopt_geometry(frame, hand_mask(frame.shape, hand), candidate) is not None:
                    data.outline = self.band.measure(frame, hand, now, hand_side=data.hand.side if data.hand else None,
                                                     hand_angles=data.hand.angles if data.hand else None)
                    data.bottle_points = data.outline.points or (None, None)
                    self.status = "AI recovered bottle; baseline kept"
                    self._lost_since = None
            if (data.outline.axis_points is None and self.baseline.distance is None and self.writer is None
                    and not self.band.joint_estimator.ready and not self.band.joint_estimator.calibrating
                    and self._lost_since is not None and now - self._lost_since >= .8):
                # Before calibration there is no reference to preserve. Do not
                # keep restricting every future search to a stale axis pose.
                self.band = None
                self.calibration = None
                self.finder.reset()
                self._lost_since = None
                self.status = 'Reacquiring bottle in current view'
        elif data.distance is None:
            self._lost_since = None
            if self.band.joint_estimator.display_only:
                self.status = 'Pink guide tracking; width reading paused'
            elif data.outline.reason.startswith('Hand lost'):
                self.status = 'Hand lost; body measurement paused'
            else:
                self.status = ('Bottle found; blue edges partly hidden' if data.outline.metrics.get('hand_occluded_profiles')
                               else 'Bottle found; body edges have low contrast')
        else:
            self._lost_since = None
            self.status = baseline_status(self.baseline) if self.baseline.distance is not None else "Keep grip relaxed for baseline"
        if self.joint_request_pending:
            if isinstance(self.band, AxisBottleTracker):
                self.band.joint_estimator.request(now)
                self.joint_request_pending = False
            else:
                self.status = 'Estimate requested; waiting for bottle cap/base'
        if (self.band is not None and data.distance is not None and self.baseline.distance is None
                and self.calibration is None and self.writer is None and now >= self.retry_after):
            self.calibration = Calibration(now + .4)
            self.calibration.ends_at = self.calibration.starts_at + 1.5
        if self.calibration is not None:
            if now >= self.calibration.ends_at:
                valid = self.band is not None and self.band.set_baseline(self.calibration.outlines)
                self.baseline = self.calibration.finish()
                self.calibration = None
                if not valid:
                    self.baseline.distance = None
                    self.retry_after = now + 1.
                elif self.band is not None:
                    data.outline = self.band.measure(frame, hand, now, hand_side=data.hand.side if data.hand else None,
                                                     hand_angles=data.hand.angles if data.hand else None)
                    data.bottle_points = data.outline.points or (None, None)
                self.history.clear()
                self.status = baseline_status(self.baseline)
            elif now >= self.calibration.starts_at:
                self.calibration.add(data)
                self.status = "Relax grip - sampling baseline"
            else:
                self.status = "Hold bottle still with grip relaxed"
        deltas, compression = self.baseline.changes(data)
        self.current_features = frame_features(data, self.band, self.baseline)
        if self.writer is not None:
            latency = (time.monotonic() - packet.captured_at) * 1000
            write_row(self.writer, now - self.record_started_at, data, self.baseline, deltas, compression,
                      self._metadata(packet, now, latency))
            if self.grip.due(now):
                self._stop_recording(completed=True)
        displayed = data.outline.metrics.get("filtered_width_reduction_pct", compression)
        mean_delta = float(np.mean(np.abs(deltas))) if deltas is not None else None
        self.history.append((now, mean_delta, displayed))
        while self.history[0][0] < now - TREND_SECONDS:
            self.history.popleft()
        detection = getattr(self.finder, 'detection', None)
        if now - getattr(self.finder, 'detection_time', -float('inf')) > .25:
            detection = None
        canvas = preview(np.zeros_like(frame), data, deltas, displayed, self.baseline,
                         self.history, now, self.status, self.writer is not None, True, self.band, detection, True)
        self.model_name = getattr(self.finder, 'model_name', self.model_name)
        reference = self.user_metadata.get('reference_grip_kgf')
        temporary = self.user_metadata.get('label_source') == 'provisional_user_assumption'
        caption = (f'Reference: {reference} kgf' + (' (TEMPORARY)' if temporary else '')
                   if reference else 'Video-only mode; no current ground-truth label')
        cv2.rectangle(canvas, (WIDTH + 1, 611), (999, 649), (25, 31, 40), -1)
        label(canvas, caption, WIDTH + 18, 634, .39, MUTED)
        cv2.rectangle(canvas, (9, 567), (632, 648), (25, 31, 40), -1)
        if self.grip.guided:
            label(canvas, self.grip.instruction(now), 18, 606, .73, PINK, 2)
            label(canvas, 'One maximal squeeze, then relax without moving the bottle away', 18, 635, .40, MUTED)
        else:
            result = self.grip.last_result or {}
            value = result.get('kgf')
            result_text = (f'Last trial (personal scale): {value:.1f} kgf' if result.get('status') == 'personal_scaling_demo'
                           else f'Last trial (research model): {value:.1f} kgf') if value is not None else self.grip.message
            label(canvas, result_text, 18, 602, .57, PINK if value is not None else MUTED)
            detail = ('Calibration reference saved; this is not a tested prediction'
                      if result.get('status') == 'personal_calibration_saved'
                      else 'G: guided trial | R: raw recording | H: new relaxed reference')
            label(canvas, detail, 18, 635, .40, MUTED)
        return {"canvas": canvas, "captured_at": packet.captured_at, "frame_id": packet.frame_id,
                "analysis_frame": frame, "model": self.model_name,
                "processing_ms": (time.perf_counter() - started) * 1000,
                "hand_detected": data.hand is not None,
                "axis_detected": data.outline.axis_points is not None,
                "baseline_ready": self.baseline.distance is not None,
                "bottle_state": 'tracked' if data.outline.axis_points is not None else ('recovering' if self.band else 'searching'),
                "status": self.status, "outline_status": data.outline.reason,
                "axis_points": data.outline.axis_points,
                "axis_quality": data.outline.axis_quality,
                "hand_occluded_profiles": data.outline.metrics.get('hand_occluded_profiles', 0),
                "bottle_search_status": self.finder.status,
                "paired_profiles": sum(v is not None for v in data.outline.band_widths),
                "left_profiles": int(np.count_nonzero(np.isfinite(self.band.visible_left))) if self.band else 0,
                "right_profiles": int(np.count_nonzero(np.isfinite(self.band.visible_right))) if self.band else 0,
                "visible_outline_segments": len(self.band.outline_segments) if self.band else 0,
                "body_outline_segments": len(self.band.body_outline_segments) if self.band else 0,
                "reference_revisions": self.band.reference_revision if self.band else 0,
                "axis_recoveries": self.band.endpoint_tracker.local_recoveries if self.band else 0,
                "joint_estimate_ready": self.band.joint_estimator.ready if self.band else False,
                "joint_estimate_status": self.band.joint_estimator.status if self.band else 'Find bottle first',
                "joint_estimate_requested": self.joint_request_pending,
                "joint_estimate_calibrating": self.band.joint_estimator.calibrating if self.band else False,
                "joint_estimate_samples": self.band.joint_estimator.stable_samples if self.band else 0,
                "joint_estimate_reason": self.band.joint_estimator.observation_reason if self.band else 'Find bottle first',
                "joint_estimate_segments": len(self.band.joint_estimator.segments) if self.band else 0,
                "joint_anchor_source": self.band.joint_estimator.anchor_source if self.band else 'none',
                "joint_display_only": self.band.joint_estimator.display_only if self.band else False,
                "joint_hand_source": self.band.joint_hand.last.source if self.band else 'missing',
                "joint_hand_age_ms": self.band.joint_hand.last.age_ms if self.band else None,
                "joint_estimated_width_px": data.outline.metrics.get('joint_estimated_width_px'),
                "joint_estimated_width_reduction_pct": data.outline.metrics.get('joint_estimated_width_reduction_pct'),
                "joint_estimated_profiles": data.outline.metrics.get('joint_estimated_profiles')}

    def close(self):
        try:
            self._stop_recording()
        finally:
            try:
                if isinstance(self.finder, AsyncBottleFinder):
                    self.finder.close()
            finally:
                self.landmarker.close()


def load_bottle_model(args):
    if args.shape_only:
        return None, "experimental_shape"
    # Full-bottle localization matters more than mask detail. On the supplied
    # occluded PET scene, YOLOX-416 preserves the full bottle under small
    # rotations while YOLO11n-seg can confidently label only the neck.
    prefer_segmentation = getattr(args, 'bottle_backend', 'auto') == 'segmentation' or args.segment_model is not None
    if not prefer_segmentation:
        try:
            path = args.bottle_model or bottle_model_file(ROOT / "models")
            detector = BottleDetector(path, threshold=.25, input_size=416)
            detector.detect(np.zeros((480, 640, 3), np.uint8))
            print('Bottle model: YOLOX at 416px (full-bottle localization)', flush=True)
            return detector, 'yolox_416_detection_only'
        except (OSError, ValueError, cv2.error) as error:
            print(f'YOLOX unavailable ({error}); trying the segmentation model.', file=sys.stderr)
    try:
        from bottle_segmentation import BottleSegmenter, model_file as segmentation_file
        path = args.segment_model or segmentation_file(ROOT / "models")
        detector = BottleSegmenter(path)
        # Warm up and validate output compatibility before selecting this model.
        detector.detect(np.zeros((480, 640, 3), np.uint8))
        print("Bottle model: YOLO11n-seg (learned instance segmentation)", flush=True)
        return detector, "yolo11n_seg"
    except (OSError, ValueError, cv2.error) as error:
        print(f"Segmentation model unavailable ({error}). Using cached YOLOX detector.", file=sys.stderr)
    try:
        path = args.bottle_model or bottle_model_file(ROOT / "models")
        print("Bottle model: YOLOX at 416px (detection only; no segmentation mask)", flush=True)
        return BottleDetector(path, threshold=.25, input_size=416), "yolox_416_detection_only"
    except (OSError, ValueError, cv2.error) as error:
        print(f"No bottle model available ({error}); live camera remains available.", file=sys.stderr)
        return None, "unavailable"


def run(capture, landmarker, output_dir: Path, video: bool, markerless: bool = False,
        detector=None, roi=None, headless=False, metadata=None,
        axis_mode=False, axis_points=None, axis_width=None, automatic=False, learned=False) -> None:
    baseline = Baseline()
    band = None
    detection = None
    auto_search = markerless and detector is not None and roi is None and axis_points is None
    automatic_active = automatic and markerless and axis_mode and roi is None and axis_points is None
    finder = (LearnedBottleFinder(detector) if learned else AutomaticBottleFinder(detector)) if automatic_active else None
    auto_baseline_started = False
    auto_retry_after = 0.0
    video_baseline_started = not markerless
    calibration = Calibration(0) if video and not markerless else None
    history = deque()
    stream = writer = None
    status = "Press B while lightly holding the bottle"
    if markerless:
        status = "Show one bottle, or P for 2 points" if auto_search else "Press P: cap center, then base center"
    if automatic_active:
        status = "Searching cap/base - show whole bottle"
    started_at = time.monotonic()
    record_started_at = 0.0
    last_timestamp_ms = -1
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) if video else 0.0
    fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    window_created = False

    try:
        if video:
            if markerless and headless and roi is None and axis_points is None and detector is None and finder is None:
                raise RuntimeError("Headless video needs --axis-points CX CY BX BY, --roi, or a bottle model.")
            stream, writer = open_recording(output_dir, markerless, metadata)
        else:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1000, 650)
            window_created = True

        while True:
            ok, frame = capture.read()
            if not ok:
                if not video or frame_index == 0:
                    raise RuntimeError("No frames received. Check the camera or video file.")
                break
            now = frame_index / fps if video else time.monotonic() - started_at
            frame_index += 1
            timestamp_ms = max(last_timestamp_ms + 1, int(now * 1000))
            last_timestamp_ms = timestamp_ms

            height, width = frame.shape[:2]
            scale = min(WIDTH / width, HEIGHT / height)
            if scale != 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
            revision = band.reference_revision if isinstance(band, AxisBottleTracker) else None
            data = observe(frame, landmarker, timestamp_ms, markerless, band)
            if isinstance(band, AxisBottleTracker) and revision is not None and revision != band.reference_revision:
                baseline.distance = None
                calibration = None
                auto_baseline_started = False
                auto_retry_after = now + .4
                if video:
                    video_baseline_started = False

            had_band = band is not None
            if markerless and frame_index == 1 and (roi is not None or axis_points is not None or (video and detector is None and finder is None)):
                if axis_points is not None:
                    band = AxisBottleTracker(frame, axis_points[:2], axis_points[2:],
                                             data.hand.points if data.hand else [], axis_width)
                elif roi is not None:
                    band = BottleBand.select(frame, tuple(roi), data.hand.points if data.hand else [])
                elif axis_mode:
                    band = select_bottle_axis(frame, data.hand, axis_width)
                else:
                    band = select_bottle_band(frame, data.hand)
                if band is None:
                    raise RuntimeError("No usable bottle body selected. Include both exterior edges and small side margins.")
            if finder is not None:
                hand_points = data.hand.points if data.hand else []
                if band is None:
                    band = finder.acquire(frame, hand_points, now)
                    status = finder.status
                elif isinstance(band, AxisBottleTracker) and data.outline.axis_points is None:
                    candidate = finder.update(frame, hand_points, now, (band.pose.cap, band.pose.base))
                    status = "Tracking lost - searching same bottle"
                    if candidate is not None:
                        accepted = band.endpoint_tracker.adopt_geometry(
                            frame, hand_mask(frame.shape, hand_points), candidate)
                        if accepted is not None:
                            data.outline = band.measure(frame, hand_points, now, hand_side=data.hand.side if data.hand else None)
                            data.bottle_points = data.outline.points or (None, None)
                            finder.reset()
                            status = "Axis recovered; original baseline kept"
                elif isinstance(band, AxisBottleTracker) and data.outline.axis_points is not None:
                    finder.reset()
            if finder is None and auto_search and band is None and frame_index % 15 == 1:
                try:
                    detections = detector.detect(frame)
                    detection = detections[0] if len(detections) == 1 else None
                    if detection is not None:
                        if axis_mode:
                            band = AxisBottleTracker.from_detection(frame, detection, data.hand.points if data.hand else [])
                        else:
                            band = BottleBand.select(frame, detection.body_region(frame.shape),
                                                     data.hand.points if data.hand else [])
                        status = "Bottle found; P to set cap/base" if band is None else "Check CAP/BASE; P correct, B baseline"
                    else:
                        status = "Multiple bottles - P to select" if detections else "Show one bottle, or P for 2 points"
                except (cv2.error, ValueError) as error:
                    print(f"Bottle detector unavailable: {error}. Press P to select cap/base.", file=sys.stderr)
                    detector = None
                    auto_search = False
                    status = "AI unavailable - P for cap/base"
                    if video and headless:
                        raise RuntimeError("Bottle detector failed. Retry with --axis-points CX CY BX BY.") from error
            if markerless and band is not None and not had_band:
                # Selection occurs after observe; do not run MediaPipe twice at
                # the same timestamp. Only update the bottle measurement.
                data.outline = (band.measure(frame, data.hand.points if data.hand else [], now,
                                            hand_side=data.hand.side if data.hand else None)
                                if isinstance(band, AxisBottleTracker) else
                                band.measure(frame, data.hand.points if data.hand else []))
                data.bottle_points = data.outline.points or (None, None)
            if video and markerless and band is not None and not video_baseline_started:
                calibration = Calibration(now)
                video_baseline_started = True
            if (automatic_active and not video and band is not None and data.distance is not None
                    and not auto_baseline_started and calibration is None and writer is None
                    and now >= auto_retry_after):
                calibration = Calibration(now + .5)
                auto_baseline_started = True

            if calibration:
                if now >= calibration.ends_at:
                    axis_baseline_ok = True
                    if isinstance(band, AxisBottleTracker):
                        axis_baseline_ok = band.set_baseline(calibration.outlines)
                    baseline = calibration.finish()
                    if not axis_baseline_ok:
                        baseline.distance = None
                        if automatic_active:
                            auto_baseline_started = False
                            auto_retry_after = now + 1.0
                    calibration = None
                    if isinstance(band, AxisBottleTracker):
                        data.outline = band.measure(frame, data.hand.points if data.hand else [], now,
                                                    hand_side=data.hand.side if data.hand else None)
                        data.bottle_points = data.outline.points or (None, None)
                    history.clear()
                    status = baseline_status(baseline)
                    print(status, flush=True)
                    if stream is not None:
                        save_recording_context(stream, band, baseline)
                elif now >= calibration.starts_at:
                    calibration.add(data)
                    status = "Hold still... sampling baseline"
                else:
                    status = f"Get ready... {math.ceil(calibration.starts_at - now)} s"

            deltas, compression = baseline.changes(data)
            if writer is not None:
                write_row(writer, now - record_started_at, data, baseline, deltas, compression, metadata)
            if video:
                continue

            mean_delta = float(np.mean(np.abs(deltas))) if deltas is not None else None
            displayed_compression = compression
            if data.outline and data.outline.metrics.get("filtered_width_reduction_pct") is not None:
                displayed_compression = data.outline.metrics["filtered_width_reduction_pct"]
            history.append((now, mean_delta, displayed_compression))
            while history[0][0] < now - TREND_SECONDS:
                history.popleft()
            cv2.imshow(WINDOW, preview(frame, data, deltas, displayed_compression, baseline,
                                       history, now, status, writer is not None, markerless, band, detection, automatic_active))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("h"), ord("H")) and isinstance(band, AxisBottleTracker) and writer is None:
                band.joint_estimator.request(now)
            if markerless and key in (ord("p"), ord("P")) and writer is None:
                selected = select_bottle_axis(frame, data.hand, axis_width)
                if selected is not None:
                    band = selected
                    baseline = Baseline()
                    history.clear()
                    calibration = None
                    auto_search = False
                    finder = None
                    automatic_active = False
                    status = "Check CAP/BASE, then B baseline"
            if markerless and key in (ord("e"), ord("E")) and writer is None:
                selected = select_bottle_band(frame, data.hand)
                if selected is not None:
                    band = selected
                    baseline = Baseline()
                    history.clear()
                    calibration = None
                    auto_search = False
                    finder = None
                    automatic_active = False
                    status = "Check both edges, then press B"
                else:
                    status = "Edges unavailable. More contrast; E retry"
            if markerless and key in (ord("a"), ord("A")) and writer is None and calibration is None:
                if automatic or axis_mode:
                    band = None
                    baseline = Baseline()
                    history.clear()
                    finder = AutomaticBottleFinder(detector)
                    automatic_active = True
                    auto_search = False
                    auto_baseline_started = False
                    status = "Searching cap/base - grip relaxed"
                elif detector is not None:
                    band = None
                    baseline = Baseline()
                    history.clear()
                    auto_search = True
                    status = "Show one upright bottle"
                else:
                    status = "AI unavailable - P for cap/base"
            if key in (ord("b"), ord("B")) and writer is None:
                if isinstance(band, AxisBottleTracker):
                    band.reset_baseline()
                baseline = Baseline()
                history.clear()
                calibration = Calibration(now + 3.0)
            joint_ready = isinstance(band, AxisBottleTracker) and band.joint_estimator.ready
            joint_sampling = isinstance(band, AxisBottleTracker) and band.joint_estimator.calibrating
            if key in (ord("r"), ord("R")) and (stream is not None or not joint_sampling):
                if stream is None:
                    if (calibration is not None or (automatic_active and baseline.distance is None)) and not joint_ready:
                        status = "Need baseline: relax BODY grip, press H"
                        continue
                    calibration = None
                    stream, writer = open_recording(output_dir, markerless, metadata)
                    save_recording_context(stream, band, baseline)
                    record_started_at = now
                else:
                    stream.close()
                    stream = writer = None
        if video and calibration is not None:
            print("Video ended during the 1-second baseline. CSV contains raw features only.")
        if video and markerless and band is None:
            if automatic_active:
                raise RuntimeError("No verified cap/body/base found automatically. Show the entire bottle and improve edge contrast. CSV contains raw hand features only.")
            raise RuntimeError("No measurable bottle found. CSV has hand features only. Retry with --axis-points CX CY BX BY.")
    finally:
        if stream is not None:
            stream.close()
            save_recording_context(stream, band, baseline)
        if window_created:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Track grip motion from a webcam or video.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    source.add_argument("--video", type=Path, help="analyze a recorded video and save a CSV")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--markerless", dest="markerless", action="store_true", help="measure bottle outline (default)")
    mode.add_argument("--markers", dest="markerless", action="store_false", help="use legacy blue/pink stickers")
    parser.set_defaults(markerless=True)
    parser.add_argument("--manual-bottle", action="store_true", help="select cap/base with P, without downloading the bottle detector")
    parser.add_argument("--shape-only", action="store_true", help="automatic cap/body/rim search without loading or downloading the bottle DNN")
    parser.add_argument("--segment-model", type=Path, help="local YOLO11n-seg ONNX model (documented 640x480 export)")
    parser.add_argument("--bottle-backend", choices=('auto', 'segmentation'), default='auto',
                        help="auto: prefer full-bottle YOLOX-416; segmentation: prefer YOLO11n-seg")
    parser.add_argument("--analysis-fps", type=float, default=12., help="maximum camera analysis rate; preview runs independently (default 12)")
    parser.add_argument("--opencv-threads", type=int, default=2, help="OpenCV worker thread limit (default 2)")
    parser.add_argument("--bottle-model", type=Path, help="local OpenCV Zoo YOLOX-S ONNX model")
    parser.add_argument("--legacy-outline", action="store_true", help="use the old horizontal body bands")
    parser.add_argument("--axis-points", type=float, nargs=4, metavar=("CX", "CY", "BX", "BY"),
                        help="cap center and base center in analysis pixels; no detector needed")
    parser.add_argument("--axis-width", type=float, help="optional initial body width in analysis pixels")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="bottle body rectangle in resized analysis pixels (max 640x480)")
    parser.add_argument("--headless", action="store_true", help="video only: never open a selection window")
    parser.add_argument("--participant", default=None, help="participant ID (default: self_temp from grip_settings.json)")
    parser.add_argument("--trial-id", default="", help="trial ID stored in CSV")
    parser.add_argument("--bottle-id", default=None, help="bottle/condition ID stored in CSV")
    parser.add_argument("--grip-kgf", type=float, help="reference dynamometer reading; not frame-level force")
    parser.add_argument("--label-source", choices=('dynamometer', 'provisional_user_assumption', 'unspecified'))
    parser.add_argument("--tested-hand", choices=('left', 'right', 'unspecified'), default=None)
    parser.add_argument("--grip-settings", type=Path, default=ROOT / 'grip_settings.json')
    parser.add_argument("--grip-model", type=Path, default=ROOT / 'models/grip_estimator.json')
    parser.add_argument("--predict-only", action='store_true', help='do not attach any current ground-truth label')
    parser.add_argument("--trial-kind", choices=('practice', 'maximal'), default='practice')
    parser.add_argument("--output-dir", type=Path, default=ROOT / "recordings", help="CSV folder")
    args = parser.parse_args()
    video = args.video is not None
    if not math.isfinite(args.analysis_fps) or not 1 <= args.analysis_fps <= 30:
        parser.error("--analysis-fps must be between 1 and 30")
    if not 1 <= args.opencv_threads <= 16:
        parser.error("--opencv-threads must be between 1 and 16")
    cv2.setNumThreads(args.opencv_threads)
    if args.headless and not video:
        parser.error("--headless requires --video")
    if args.axis_points is not None and args.roi is not None:
        parser.error("use either --axis-points or --roi")
    if args.axis_width is not None and (not math.isfinite(args.axis_width) or args.axis_width <= 20):
        parser.error("--axis-width must be finite and greater than 20 pixels")
    if args.grip_kgf is not None and (not math.isfinite(args.grip_kgf) or args.grip_kgf <= 0):
        parser.error("--grip-kgf must be a finite positive value")
    if args.label_source == 'dynamometer' and args.grip_kgf is None:
        parser.error('dynamometer labels require an explicit --grip-kgf reading')
    try:
        grip_settings = load_settings(args.grip_settings)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    metadata = {**label_metadata(grip_settings, args.participant, args.grip_kgf, args.label_source,
                                 args.bottle_id, args.tested_hand, args.predict_only),
                'trial_id': args.trial_id, 'trial_kind': args.trial_kind}
    automatic = (args.markerless and not args.manual_bottle and not args.legacy_outline
                 and args.roi is None and args.axis_points is None)
    capture = cv2.VideoCapture(str(args.video) if video else args.camera)
    try:
        if not capture.isOpened():
            source_name = str(args.video) if video else f"camera {args.camera}"
            raise RuntimeError(f"Cannot open {source_name}. See the troubleshooting in README.md.")
        if not video:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not video and automatic:
            from realtime import run_live

            def session_factory():
                landmarker = LiveHandModels()

                def search_factory():
                    detector, model_name = load_bottle_model(args)
                    return BottleSearchProcessor(detector, model_name, args.shape_only)

                return RealtimeSession(landmarker, None, args.output_dir.resolve(),
                                       metadata, 'loading', args.shape_only, search_factory,
                                       grip_settings=grip_settings, grip_model_path=args.grip_model)

            run_live(capture, session_factory, WINDOW, ROOT / "debug", args.analysis_fps)
            return 0
        detector = None
        if automatic and not args.shape_only:
            detector, name = load_bottle_model(args)
            metadata["bottle_model"] = name
        elif args.markerless and not args.manual_bottle and not args.shape_only and args.roi is None and args.axis_points is None:
            try:
                detector_path = args.bottle_model or bottle_model_file(ROOT / "models")
                detector = BottleDetector(detector_path)
            except (OSError, cv2.error, ValueError) as error:
                print(f"Bottle AI unavailable ({error}). Automatic shape search remains available.", file=sys.stderr)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_file())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            run(capture, landmarker, args.output_dir.resolve(), video, args.markerless,
                detector, args.roi, args.headless, metadata,
                not args.legacy_outline, args.axis_points, args.axis_width, automatic,
                automatic and not args.shape_only)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
