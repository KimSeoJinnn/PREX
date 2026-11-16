"""Partial occlusion must not erase visible edges or invent the hidden width."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bottle_axis import AxisBottleTracker, trace_contour
from bottle_detector import Detection
from bottle_learned import LearnedBottleFinder
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH


class PartialContourTests(unittest.TestCase):
    def test_hand_gap_cannot_pull_the_visible_contour_above_it(self):
        gradient = np.zeros((18, 64), np.float32)
        gradient[:7, 10] = 60
        gradient[9:, 42] = 60
        blocked = np.zeros(gradient.shape, np.uint8)
        blocked[7:9] = 255
        path = trace_contour(gradient, blocked, np.full(18, 26.), np.full(18, 30.))
        np.testing.assert_allclose(path[:7], 10.)
        np.testing.assert_allclose(path[9:], 42.)
        self.assertTrue(np.isnan(path[7:9]).all())

    def test_fully_hidden_and_weak_runs_never_produce_a_contour(self):
        gradient = np.zeros((20, 64), np.float32)
        gradient[5:10, 25] = 100  # Strong, but covered by the hand.
        blocked = np.full(gradient.shape, 255, np.uint8)
        blocked[12:] = 0  # Visible background has no edge.
        path = trace_contour(gradient, blocked, np.full(20, 26.), np.full(20, 20.))
        self.assertTrue(np.isnan(path).all())

    def test_single_visible_side_follows_progressive_dent_without_creating_width(self):
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertTrue(tracker.set_baseline([tracker.last] * 4))
        baseline = tracker.baseline_left.copy(), tracker.baseline_right.copy()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[:, 365:430] = 255  # Only the right contour is hidden.
        station = int(np.argmin(abs(CAP[1] + tracker.positions * (BASE[1] - CAP[1]) - 265)))
        y = CAP[1] + tracker.positions[station] * (BASE[1] - CAP[1])
        for tick, dent in enumerate((.12, .24, .36, .48, .56), start=1):
            current, pose = axis_frame(dent=dent, asymmetric=True)
            current[np.all(current > 200, axis=2)] = 165  # Remove internal highlight.
            with patch('bottle_axis.hand_mask', return_value=blocked), patch.object(
                    tracker.endpoint_tracker, 'track', return_value=(pose, 1., 'Axis OK')):
                measured = tracker.measure(current, [], float(tick))
            x = tracker.visible_left[station]
            self.assertTrue(np.isfinite(x), f'Visible left contour lost at dent {dent}')
            observed = tracker.pose.image_points([[x, 32 + tracker.positions[station] * 320]])[0, 0]
            expected = 320 - 65 + 130 * dent * np.exp(-.5 * ((y - 265) / 30) ** 2)
            self.assertAlmostEqual(observed, expected, delta=3.)
            self.assertTrue(np.isnan(tracker.visible_right[station]))
            self.assertIsNone(measured.width_px)
            self.assertIsNone(measured.metrics['median_width_reduction_pct'])
        np.testing.assert_allclose(tracker.baseline_left, baseline[0], equal_nan=True)
        np.testing.assert_allclose(tracker.baseline_right, baseline[1], equal_nan=True)

    @staticmethod
    def finder():
        detections = iter([[Detection((255, 48, 130, 371), .9)],
                           [Detection((345, 48, 130, 371), .9)]])
        return LearnedBottleFinder(SimpleNamespace(detect=lambda _: next(detections)))

    def test_initial_confirmation_accepts_same_observed_bottle_when_flow_fails(self):
        finder = self.finder()
        first, _ = axis_frame()
        moved, expected = axis_frame(shift=(90., 0.))
        self.assertIsNone(finder.update(first, [], 0.))
        first_base = finder.candidate.base.copy()
        with patch('bottle_learned.EndpointTracker.track', return_value=(None, 0., 'no flow')):
            result = finder.update(moved, [], 2.)
        self.assertIsNotNone(result, finder.status)
        np.testing.assert_allclose(result.cap, expected.cap, atol=4.)
        # The observed ellipse rim is below its geometric center. Check that
        # the same physical rim follows the translation, not its center seed.
        np.testing.assert_allclose(result.base, first_base + (90., 0.), atol=1.5)

    def test_initial_confirmation_rejects_different_cap_when_flow_fails(self):
        finder = self.finder()
        first, _ = axis_frame()
        moved, _ = axis_frame(shift=(90., 0.))
        moved[48:83, 375:446] = (220, 70, 30)
        self.assertIsNone(finder.update(first, [], 0.))
        with patch('bottle_learned.EndpointTracker.track', return_value=(None, 0., 'no flow')):
            self.assertIsNone(finder.update(moved, [], 2.))
        self.assertEqual(finder.count, 1)


if __name__ == '__main__':
    unittest.main()
