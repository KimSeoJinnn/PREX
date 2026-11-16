"""Regression cases for transparent interiors and incomplete visible outlines."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_detector import Detection
from bottle_learned import LearnedBottleFinder
from main import Calibration, FrameData, RealtimeSession
from realtime import FramePacket
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH


class DetectionRepairTests(unittest.TestCase):
    def test_stationary_background_cannot_anchor_a_moving_transparent_base(self):
        rng = np.random.default_rng(5)
        background = np.full((480, 640, 3), 165, np.uint8)
        for _ in range(900):
            x, y = rng.integers((180, 10), (460, 475))
            color = int(rng.integers(30, 240))
            cv2.rectangle(background, (int(x), int(y)), (int(x + 3), int(y + 3)), (color,) * 3, -1)

        def scene(angle, shift):
            foreground, pose = axis_frame(angle=angle, shift=(shift, 0.))
            result = background.copy()
            mask = np.any(foreground != 165, axis=2)
            result[mask] = foreground[mask]
            return result, pose

        frame, _ = scene(0, 0)
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        tracker.endpoint_tracker.allow_geometry = False
        moved, expected = scene(1, 4)
        result = tracker.measure(moved, [], .1)
        if result.axis_points is not None:
            # Correct image-supported tracking is allowed; background-locked
            # measurements (previously ~7px base error after one step) are not.
            self.assertLess(np.linalg.norm(tracker.pose.base - expected.base), 3.)
        else:
            self.assertIn('motion disagrees', result.reason)
            self.assertIsNone(result.width_px)
            self.assertEqual(result.metrics, {})

    def test_same_moving_bottle_can_finish_automatic_confirmation(self):
        regions = iter([Detection((255, 48, 130, 371), .9), Detection((310, 48, 130, 371), .9)])
        finder = LearnedBottleFinder(SimpleNamespace(detect=lambda _: [next(regions)]))
        first, _ = axis_frame()
        second, pose = axis_frame(shift=(55, 0))
        self.assertIsNone(finder.update(first, [], 0))
        result = finder.update(second, [], 2)
        self.assertIsNotNone(result, finder.status)
        np.testing.assert_allclose(result.cap, pose.cap, atol=6)
        np.testing.assert_allclose(result.base, pose.base, atol=10)

    def test_single_background_bottle_far_from_hand_is_not_the_gripped_target(self):
        detector = SimpleNamespace(detect=lambda _: [Detection((40, 60, 100, 320), .9)])
        finder = LearnedBottleFinder(detector, stable_samples=1)
        self.assertIsNone(finder.update(np.zeros((480, 640, 3), np.uint8), [(480, 260)] * 21, 0))
        self.assertIsNone(finder.detection)

    def test_short_visible_runs_are_displayed_without_inventing_width(self):
        frame, _ = axis_frame()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[145:380] = 255
        blocked[229:233] = 0
        with patch('bottle_axis.hand_mask', return_value=blocked):
            tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertIsNone(tracker.last.width_px)
        self.assertGreater(len(tracker.body_outline_segments), 0)
        self.assertFalse(tracker.set_baseline([tracker.last] * 4))
        canvas = np.zeros_like(frame)
        tracker.draw(canvas)
        # Bottle blue remains distinct from the hand's green, including
        # visible runs too short to support a reliable width statistic.
        blue = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] > 150) & (canvas[:, :, 2] < 120)
        self.assertTrue(np.any(blue[229:233, 250:265]))
        self.assertFalse(np.any(blue & (blocked > 0)))

    def test_lost_axis_clears_dense_display_instead_of_leaving_old_blue_lines(self):
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertTrue(tracker.body_outline_segments)
        tracker.endpoint_tracker.allow_geometry = False
        result = tracker.measure(np.full_like(frame, 165), [], .1)
        self.assertIsNone(result.axis_points)
        self.assertEqual(tracker.body_outline_segments, [])

    def test_reference_with_already_present_undetected_hand_is_discarded(self):
        frame, _ = axis_frame()
        cv2.rectangle(frame, (250, 205), (300, 300), (70, 120, 190), -1)
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertTrue(tracker.set_baseline([tracker.last] * 4))
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[200:305, 245:305] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked), \
                patch('bottle_axis.body_occlusion_mask', return_value=blocked):
            tracker.measure(frame, [(280, 250)] * 21, .1)
        self.assertEqual(tracker.reference_revision, 1)
        self.assertIsNone(tracker.baseline_left)
        self.assertIsNone(tracker.last.metrics['median_width_reduction_pct'])

    def test_valid_unheld_reference_is_kept_when_new_hand_enters_later(self):
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertTrue(tracker.set_baseline([tracker.last] * 4))
        baseline = tracker.baseline_left.copy()
        new = frame.copy()
        cv2.rectangle(new, (250, 205), (300, 300), (30, 60, 90), -1)
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[210:295, 255:295] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked), \
                patch('bottle_axis.body_occlusion_mask', return_value=blocked):
            tracker.measure(new, [(280, 250)] * 21, .1)
        self.assertEqual(tracker.reference_revision, 0)
        np.testing.assert_allclose(tracker.baseline_left, baseline, equal_nan=True)

    def test_reference_revision_cancels_the_session_calibration_with_old_samples(self):
        frame, _ = axis_frame()
        model = SimpleNamespace(close=lambda: None)
        session = RealtimeSession(model, None, Path('unused'), {}, 'test')
        session.band = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        session.baseline.distance = 130.
        session.calibration = Calibration(0.)
        session.calibration.add(FrameData(None, (None, None), session.band.last))

        def observe(*_):
            session.band.reference_revision += 1
            session.band.reset_baseline()
            return FrameData(None, (None, None), session.band.last)

        try:
            with patch('main.observe', side_effect=observe):
                session.process(FramePacket(0, 100., frame), [])
            self.assertIsNone(session.baseline.distance)
            self.assertIsNone(session.calibration)
        finally:
            session.close()

    def test_plain_white_cap_keeps_its_boundary_features_for_tracking(self):
        frame, _ = axis_frame()
        cv2.rectangle(frame, (285, 48), (355, 82), (240, 240, 240), -1)
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        tracker.endpoint_tracker.allow_geometry = False
        for index in range(1, 4):
            matrix = cv2.getRotationMatrix2D((320., 237.5), index, 1.)
            matrix[:, 2] += (index * 2., 0.)
            moved = cv2.warpAffine(frame, matrix, (640, 480), borderValue=(165, 165, 165))
            expected = cv2.transform(np.array([[CAP, BASE]], np.float32), matrix)[0]
            result = tracker.measure(moved, [], index / 12)
            self.assertIsNotNone(result.axis_points, result.reason)
            np.testing.assert_allclose(result.axis_points, expected, atol=3.)


if __name__ == '__main__':
    unittest.main()
