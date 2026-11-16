import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker, BottlePose, propose_endpoints
from bottle_detector import Detection
from main import run


CAP = np.array([320., 65.])
BASE = np.array([320., 410.])
BODY_WIDTH = 130


def axis_frame(dent=0., asymmetric=False, angle=0., scale=1., shift=(0., 0.)):
    frame = np.full((480, 640, 3), 165, np.uint8)
    left, right = [], []
    for y in range(112, 411):
        half = min(65, 25 + (y - 112) * 1.5)
        local = np.exp(-.5 * ((y - 265) / 30) ** 2)
        dl = BODY_WIDTH * dent * local * (1 if asymmetric else .5)
        dr = BODY_WIDTH * dent * local * (0 if asymmetric else .5)
        left.append((round(320 - half + dl), y))
        right.append((round(320 + half - dr), y))
    cv2.polylines(frame, [np.array(left), np.array(right)], False, (130, 130, 130), 2, cv2.LINE_AA)
    cv2.line(frame, (302, 84), (295, 112), (120, 120, 120), 2)
    cv2.line(frame, (338, 84), (345, 112), (120, 120, 120), 2)
    cv2.rectangle(frame, (285, 48), (355, 82), (75, 105, 135), -1)
    for x in range(289, 354, 7):
        cv2.line(frame, (x, 50), (x, 80), (110, 145, 175), 2)
    cv2.putText(frame, "C", (310, 74), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)
    cv2.ellipse(frame, (320, 409), (65, 10), 0, 0, 360, (110, 110, 110), 2, cv2.LINE_AA)
    for x, y in ((292, 400), (314, 406), (340, 401), (350, 412)):
        cv2.circle(frame, (x, y), 2, (125, 125, 125), -1)
    # Internal highlight remains much stronger than the transparent outline.
    cv2.line(frame, (310, 145), (310, 325), (225, 225, 225), 2)
    transform = cv2.getRotationMatrix2D((320., 237.5), angle, scale)
    transform[:, 2] += shift
    moved = cv2.warpAffine(frame, transform, (640, 480), borderValue=(165, 165, 165))
    endpoints = cv2.transform(np.array([[CAP, BASE]], np.float32), transform)[0]
    return moved, BottlePose(endpoints[0].astype(float), endpoints[1].astype(float))


class AxisTests(unittest.TestCase):
    def tracker(self):
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertIsNotNone(tracker.last.width_px)
        self.assertTrue(tracker.set_baseline([tracker.measure(frame, [], i / 30) for i in range(4)]))
        return tracker

    def test_two_points_rectify_rotation_and_distance_without_false_deformation(self):
        tracker = self.tracker()
        for angle, scale, shift in ((18, 1., (10., 0.)), (-22, .85, (-20., 8.)), (0, 1.1, (0., -4.))):
            frame, pose = axis_frame(angle=angle, scale=scale, shift=shift)
            with patch.object(tracker.endpoint_tracker, "track", return_value=(pose, 1., "Axis OK")):
                measured = tracker.measure(frame, [], 1.)
            self.assertIsNotNone(measured.width_px)
            self.assertAlmostEqual(measured.metrics["median_width_reduction_pct"], 0, delta=2)
            self.assertAlmostEqual(measured.metrics["visible_area_reduction_pct"], 0, delta=2)

    def test_local_dent_is_detected_when_median_width_barely_changes(self):
        tracker = self.tracker()
        frame, pose = axis_frame(dent=.24)
        with patch.object(tracker.endpoint_tracker, "track", return_value=(pose, 1., "Axis OK")):
            measured = tracker.measure(frame, [], 1.)
        metrics = measured.metrics
        self.assertGreater(metrics["dent_p95_pct"], 17)
        self.assertLess(metrics["dent_p95_pct"], 29)
        self.assertGreater(metrics["dent_p95_pct"], metrics["median_width_reduction_pct"] + 8)
        self.assertGreater(metrics["visible_area_reduction_pct"], 3)

    def test_asymmetric_dent_moves_one_contour(self):
        tracker = self.tracker()
        frame, pose = axis_frame(dent=.20, asymmetric=True)
        with patch.object(tracker.endpoint_tracker, "track", return_value=(pose, 1., "Axis OK")):
            measured = tracker.measure(frame, [], 1.)
        self.assertGreater(measured.metrics["left_dent_p95_pct"], 13)
        self.assertLess(measured.metrics["right_dent_p95_pct"], 3)
        self.assertGreater(measured.metrics["center_shift_rms_pct"], 2)

    def test_real_endpoint_optical_flow_follows_motion(self):
        tracker = self.tracker()
        for i in range(1, 6):
            frame, expected = axis_frame(angle=i * 1.5, scale=1 - i * .008, shift=(i * 2., 0.))
            measured = tracker.measure(frame, [], i / 10)
            self.assertIsNotNone(measured.axis_points, measured.reason)
            np.testing.assert_allclose(tracker.pose.cap, expected.cap, atol=3)
            np.testing.assert_allclose(tracker.pose.base, expected.base, atol=3)
            self.assertAlmostEqual(measured.metrics["median_width_reduction_pct"], 0, delta=3)

    def test_lost_endpoint_never_returns_previous_axis_or_deformation(self):
        tracker = self.tracker()
        blank = np.full((480, 640, 3), 165, np.uint8)
        measured = tracker.measure(blank, [], 1.)
        self.assertIsNone(measured.axis_points)
        self.assertIsNone(measured.width_px)
        self.assertEqual(measured.metrics, {})

    def test_baseline_reset_requires_new_samples_and_uses_new_rest_shape(self):
        tracker = self.tracker()
        frame, pose = axis_frame(dent=.18)
        with patch.object(tracker.endpoint_tracker, "track", return_value=(pose, 1., "Axis OK")):
            self.assertGreater(tracker.measure(frame, [], 1.).metrics["dent_p95_pct"], 10)
            tracker.reset_baseline()
            self.assertIsNone(tracker.measure(frame, [], 2.).metrics["median_width_reduction_pct"])
            self.assertFalse(tracker.set_baseline([tracker.last]))
            samples = [tracker.measure(frame, [], 3. + i / 30) for i in range(4)]
            self.assertTrue(tracker.set_baseline(samples))
            measured = tracker.measure(frame, [], 4.)
            self.assertAlmostEqual(measured.metrics["median_width_reduction_pct"], 0, delta=.5)
            self.assertAlmostEqual(measured.metrics["dent_p95_pct"], 0, delta=.5)

    def test_occluded_body_is_missing_and_area_is_not_invented(self):
        tracker = self.tracker()
        frame, _ = axis_frame()
        blocked = np.zeros((480, 640), np.uint8)
        blocked[190:310] = 255
        with patch("bottle_axis.hand_mask", return_value=blocked):
            measured = tracker.measure(frame, [], 1.)
        self.assertTrue(any(value is None for value in measured.band_widths))
        self.assertIsNone(measured.metrics["visible_area_reduction_pct"])
        self.assertLess(measured.metrics["area_coverage"], .6)
        self.assertAlmostEqual(measured.metrics["median_width_reduction_pct"], 0, delta=1)

    def test_invalid_or_repeated_endpoints_are_rejected(self):
        frame, _ = axis_frame()
        for cap, base in (((320, 60), (320, 60)), ((0, 60), BASE), ((np.nan, 60), BASE)):
            with self.assertRaises(ValueError):
                AxisBottleTracker(frame, cap, base, [])

    def test_cap_and_base_proposals_use_image_details_in_tilted_bottle(self):
        frame, expected = axis_frame(angle=16)
        changed = np.any(np.abs(frame.astype(int) - 165) > 8, axis=2).astype(np.uint8)
        region = cv2.boundingRect(cv2.findNonZero(changed))
        cap, base, source = propose_endpoints(frame, Detection(region, .9), [])
        np.testing.assert_allclose(cap, expected.cap, atol=7)
        np.testing.assert_allclose(base, expected.base, atol=10)
        self.assertEqual(source, "detector_image_proposal")

    def test_axis_csv_and_baseline_curves_survive_video_pipeline(self):
        frames = iter((True, axis_frame(dent=0 if i < 24 else .2)[0]) for i in range(40))
        capture = SimpleNamespace(get=lambda _: 20., read=lambda: next(frames, (False, None)))
        landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(hand_landmarks=[]))
        with tempfile.TemporaryDirectory() as directory:
            run(capture, landmarker, Path(directory), video=True, markerless=True,
                headless=True, axis_mode=True, axis_points=(*CAP, *BASE), axis_width=BODY_WIDTH)
            path = next(Path(directory).glob("*.csv"))
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(None not in row for row in rows))
            self.assertTrue(all(row["dent_p95_pct"] == "" for row in rows[:20]))
            self.assertGreater(float(rows[-1]["dent_p95_pct"]), 14)
            self.assertTrue(rows[-1]["cap_x_px"])
            context = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(context["bottle_method"], "cap_base_aligned_contour_v1")
            self.assertEqual(len(context["baseline_left"]), 31)


if __name__ == "__main__":
    unittest.main()
