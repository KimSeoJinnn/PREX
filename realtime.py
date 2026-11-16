"""Latest-frame capture and analysis, independent of the GUI thread."""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_at: float
    frame: np.ndarray


def fit_frame(frame):
    height, width = frame.shape[:2]
    scale = min(640 / width, 480 / height)
    return cv2.resize(frame, (round(width * scale), round(height * scale))) if scale != 1 else frame


class LatestCapture:
    def __init__(self, capture):
        self.capture = capture
        self._lock = Lock()
        self._stop = Event()
        self.latest = None
        self.error = None
        self.thread = Thread(target=self._run, name="grip-camera", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        frame_id, failures = 0, 0
        try:
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    failures += 1
                    if failures >= 5:
                        raise RuntimeError("Camera stopped delivering frames")
                    self._stop.wait(.04)
                    continue
                failures = 0
                captured_at = time.monotonic()
                packet = FramePacket(frame_id, captured_at, fit_frame(frame))
                with self._lock:
                    self.latest = packet
                frame_id += 1
        except Exception as error:
            self.error = error

    def read_latest(self):
        with self._lock:
            return self.latest

    def close(self):
        self._stop.set()
        self.thread.join(timeout=2)


class LatestWorker:
    def __init__(self, factory, max_fps=12):
        self.factory, self.max_fps = factory, max_fps
        self._condition = Condition()
        self._stop = False
        self._pending = None
        self._commands = deque()
        self._latest = None
        self._submitted = -1
        self.error = None
        self.loading = True
        self.dropped = 0
        self.thread = Thread(target=self._run, name="grip-analysis", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, packet):
        with self._condition:
            if packet.frame_id <= self._submitted:
                return
            self._submitted = packet.frame_id
            if self._pending is not None:
                self.dropped += 1
            self._pending = packet
            self._condition.notify_all()

    def command(self, key):
        with self._condition:
            self._commands.append(key)
            self._condition.notify_all()

    def read_latest(self):
        with self._condition:
            return self._latest

    def _run(self):
        processor = None
        try:
            processor = self.factory()
            self.loading = False
            next_at = 0.
            while True:
                with self._condition:
                    while not self._stop and (self._pending is None or time.monotonic() < next_at):
                        timeout = max(.001, next_at - time.monotonic()) if self._pending is not None else .1
                        self._condition.wait(timeout=timeout)
                    if self._stop:
                        break
                    packet, self._pending = self._pending, None
                    commands, self._commands = list(self._commands), deque()
                started = time.monotonic()
                result = processor.process(packet, commands)
                with self._condition:
                    if result is not None:
                        self._latest = result
                next_at = started + 1 / self.max_fps
        except Exception as error:
            self.error = error
        finally:
            self.loading = False
            if processor is not None:
                try:
                    processor.close()
                except Exception as error:
                    self.error = self.error or error

    def close(self):
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        # A normal analysis tick must finish and close its recording stream.
        # Model initialization can involve a download; don't trap a closing UI.
        self.thread.join(timeout=2 if self.loading else 8)


def compose_preview(packet, result, now, preview_fps=0., loading=False):
    if result is None:
        canvas = np.full((650, 1000, 3), (25, 31, 40), np.uint8)
    else:
        canvas = result["canvas"].copy()
    frame = packet.frame
    h, w = frame.shape[:2]
    canvas[:480, :640] = (25, 31, 40)
    canvas[:h, :w] = frame
    age = None if result is None else max(0., now - result["captured_at"])
    # Old overlays never stick to a new live image. Measurements remain labeled
    # with their original capture time, and no CSV row is written by this GUI.
    if result is not None and age <= .25:
        overlay = result["canvas"][:h, :w]
        # Same non-black-pixel rule, without allocating/indexing large NumPy
        # boolean arrays on every preview frame. Keep the analysis CPU free.
        ink = cv2.bitwise_not(cv2.inRange(overlay, (0, 0, 0), (0, 0, 0)))
        cv2.copyTo(overlay, ink, canvas[:h, :w])
    status = "Loading AI models; live camera continues" if loading else "Waiting for analysis"
    if result is not None:
        status = f"Preview {preview_fps:.0f} fps | Analysis {result['processing_ms']:.0f} ms | Age {age * 1000:.0f} ms"
    cv2.rectangle(canvas, (0, 0), (640, 29), (25, 31, 40), -1)
    cv2.putText(canvas, status, (9, 20), cv2.FONT_HERSHEY_SIMPLEX, .46, (230, 230, 230), 1, cv2.LINE_AA)
    if age is not None and age > .5:
        cv2.putText(canvas, "Values at right are from an older analysis", (9, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, .44, (90, 170, 255), 1, cv2.LINE_AA)
    return canvas


def save_debug_snapshot(debug_dir, packet, result, now, preview_fps=0., loading=False):
    """Save matching input, preview, and measured-state evidence on D only."""
    from datetime import datetime
    import json

    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime('scene_%Y%m%d_%H%M%S_%f')
    raw = debug_dir / f'{stem}.png'
    cv2.imwrite(str(raw), packet.frame)
    cv2.imwrite(str(debug_dir / f'{stem}_preview.png'), compose_preview(packet, result, now, preview_fps, loading))
    fields = ('frame_id', 'captured_at', 'model', 'processing_ms', 'hand_detected', 'axis_detected',
              'baseline_ready', 'bottle_state', 'status', 'outline_status', 'axis_points',
              'axis_quality', 'hand_occluded_profiles', 'bottle_search_status',
              'paired_profiles', 'left_profiles', 'right_profiles', 'visible_outline_segments',
              'body_outline_segments', 'reference_revisions', 'axis_recoveries',
              'joint_estimate_ready', 'joint_estimate_status', 'joint_estimated_width_px',
              'joint_estimate_requested', 'joint_estimate_calibrating', 'joint_estimate_samples',
              'joint_estimate_reason', 'joint_estimate_segments',
              'joint_anchor_source', 'joint_display_only', 'joint_hand_source', 'joint_hand_age_ms',
              'joint_estimated_width_reduction_pct', 'joint_estimated_profiles')
    analysis = None if result is None else {key: result[key] for key in fields if key in result}
    if result is not None and result.get('analysis_frame') is not None:
        cv2.imwrite(str(debug_dir / f'{stem}_analysis.png'), result['analysis_frame'])
    state = {'camera_frame_id': packet.frame_id, 'camera_captured_at': packet.captured_at,
             'analysis_age_ms': None if result is None else max(0., now - result['captured_at']) * 1000,
             'loading': loading, 'analysis': analysis}
    (debug_dir / f'{stem}.json').write_text(json.dumps(state, indent=2, allow_nan=False), encoding='utf-8')
    return raw


def run_live(capture, factory, window, debug_dir, max_fps=12):
    camera = LatestCapture(capture).start()
    worker = LatestWorker(factory, max_fps).start()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1000, 650)
    last_id, preview_fps, previous = -1, 0., time.monotonic()
    try:
        while True:
            now = time.monotonic()
            packet = camera.read_latest()
            result = worker.read_latest()
            if packet is not None:
                if packet.frame_id != last_id:
                    elapsed = now - previous
                    preview_fps = .85 * preview_fps + .15 / max(elapsed, .001)
                    previous, last_id = now, packet.frame_id
                    worker.submit(packet)
                cv2.imshow(window, compose_preview(packet, result, now, preview_fps, worker.loading))
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('a'), ord('A'), ord('b'), ord('B'), ord('h'), ord('H'), ord('r'), ord('R'), ord('g'), ord('G')):
                worker.command(chr(key).lower())
            if key in (ord('d'), ord('D')) and packet is not None:
                path = save_debug_snapshot(debug_dir, packet, result, now, preview_fps, worker.loading)
                print(f'Debug snapshot (image, preview, analysis, JSON): {path}', flush=True)
            if camera.error is not None:
                raise RuntimeError(str(camera.error))
            if worker.error is not None:
                raise RuntimeError(f"Analysis stopped: {worker.error}")
    finally:
        camera.close()
        worker.close()
        cv2.destroyAllWindows()
