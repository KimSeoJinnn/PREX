import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bottle_auto import AutomaticBottleFinder
from bottle_axis import AxisBottleTracker
from bottle_landmarks import detect_landmarks
from main import run
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH


class AutomaticTests(unittest.TestCase):
    def test_both_centers_are_found_without_model_or_input_points(self):
        for angle in (0, 12, -15):
            frame, expected = axis_frame(angle=angle)
            found = detect_landmarks(frame, [])
            self.assertEqual(len(found), 1)
            np.testing.assert_allclose(found[0].cap, expected.cap, atol=5)
            np.testing.assert_allclose(found[0].base, expected.base, atol=6)

    def test_white_cap_does_not_require_color_marker(self):
        frame, expected = axis_frame()
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        found = detect_landmarks(frame, [])
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0].cap, expected.cap, atol=5)
        np.testing.assert_allclose(found[0].base, expected.base, atol=6)

    def test_rectangle_and_empty_background_are_rejected(self):
        blank = np.full((480, 640, 3), 165, np.uint8)
        rectangle = blank.copy()
        cv2.rectangle(rectangle, (270, 50), (370, 420), (75, 105, 135), 3)
        self.assertEqual(detect_landmarks(blank, []), [])
        self.assertEqual(detect_landmarks(rectangle, []), [])

    def test_waits_for_stable_observations_then_acquires(self):
        finder = AutomaticBottleFinder(interval=.2)
        frame, _ = axis_frame()
        self.assertIsNone(finder.acquire(frame, [], 0))
        self.assertIsNone(finder.acquire(frame, [], .1))
        self.assertIsNone(finder.acquire(frame, [], .3))
        tracker = finder.acquire(frame, [], .6)
        self.assertIsNotNone(tracker)
        self.assertIsNotNone(tracker.last.width_px)
        self.assertEqual(tracker.source, "automatic_shape")

    def test_jumping_candidate_cannot_lock(self):
        finder = AutomaticBottleFinder(interval=.2)
        for i in range(5):
            frame, _ = axis_frame(shift=((i % 2) * 45, 0))
            self.assertIsNone(finder.acquire(frame, [], i * .3))

    def test_multiple_bottles_do_not_silently_select_one(self):
        first, _ = axis_frame(shift=(-150, 0))
        second, _ = axis_frame(shift=(150, 0))
        mask = np.any(second != 165, axis=2)
        first[mask] = second[mask]
        finder = AutomaticBottleFinder()
        for i in range(3):
            self.assertIsNone(finder.acquire(first, [], i * .4))
        self.assertIn("Multiple", finder.status)

    def test_optional_model_failure_keeps_automatic_shape_search(self):
        detector = SimpleNamespace(detect=lambda _: (_ for _ in ()).throw(ValueError("bad model")))
        finder = AutomaticBottleFinder(detector)
        frame, _ = axis_frame()
        tracker = None
        for i in range(3):
            tracker = finder.acquire(frame, [], i * .4)
        self.assertIsNotNone(tracker)
        self.assertEqual(finder.model_error, "bad model")

    def test_texture_tracking_failure_uses_observed_rim_and_keeps_baseline(self):
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertTrue(tracker.set_baseline([tracker.measure(frame, []) for _ in range(3)]))
        reference = tracker.baseline_left.copy()
        moved, expected = axis_frame(angle=3, shift=(6, 0))
        with patch("bottle_axis.cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)), \
                patch.object(tracker.endpoint_tracker, "_template_fallback", return_value=None):
            measured = tracker.measure(moved, [], 1.)
        self.assertIsNotNone(measured.axis_points, measured.reason)
        np.testing.assert_allclose(tracker.pose.cap, expected.cap, atol=5)
        np.testing.assert_allclose(tracker.pose.base, expected.base, atol=6)
        np.testing.assert_array_equal(tracker.baseline_left, reference)
        self.assertAlmostEqual(measured.metrics["median_width_reduction_pct"], 0, delta=2)

    def test_automatic_video_never_opens_selection_and_saves_valid_deformation(self):
        frames = iter((True, axis_frame(dent=0 if i < 46 else .2)[0]) for i in range(58))
        capture = SimpleNamespace(get=lambda _: 20., read=lambda: next(frames, (False, None)))
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        with tempfile.TemporaryDirectory() as directory, \
                patch("main.select_bottle_axis", side_effect=AssertionError("No clicks allowed")), \
                patch("main.select_bottle_band", side_effect=AssertionError("No ROI allowed")):
            run(capture, landmarker, Path(directory), video=True, markerless=True,
                headless=True, axis_mode=True, automatic=True)
            path = next(Path(directory).glob("*.csv"))
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 58)
            self.assertTrue(all(None not in row for row in rows))
            self.assertEqual(rows[0]["cap_x_px"], "")
            self.assertTrue(rows[-1]["cap_x_px"])
            self.assertGreater(float(rows[-1]["dent_p95_pct"]), 14)
            context = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(context["axis_seed_source"], "automatic_shape")
            self.assertIsNotNone(context["baseline_width_px"])

    def test_camera_automatically_calibrates_and_ignores_record_until_ready(self):
        frame, _ = axis_frame()
        capture = SimpleNamespace(read=lambda: (True, frame.copy()))
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        ticks = iter(i * .25 for i in range(100))
        keys = iter([ord('r')] + [0] * 12 + [ord('r')] + [0] * 2 + [ord('r'), ord('q')])
        with tempfile.TemporaryDirectory() as directory, \
                patch("main.time.monotonic", side_effect=lambda: next(ticks)), \
                patch("main.cv2.namedWindow"), patch("main.cv2.resizeWindow"), \
                patch("main.cv2.imshow"), patch("main.cv2.destroyAllWindows"), \
                patch("main.cv2.getWindowProperty", return_value=1), \
                patch("main.cv2.waitKey", side_effect=lambda _: next(keys)):
            run(capture, landmarker, Path(directory), video=False, markerless=True,
                axis_mode=True, automatic=True)
            paths = list(Path(directory).glob("*.csv"))
            self.assertEqual(len(paths), 1)
            with paths[0].open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(row["bottle_baseline_width_px"] for row in rows))
            self.assertTrue(all(row["cap_x_px"] for row in rows))


if __name__ == "__main__":
    unittest.main()
