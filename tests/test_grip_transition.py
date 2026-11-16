"""A grip change must not trap bottle recovery behind failed optical flow."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_endpoints import cap_identity, observe_endpoints
from bottle_landmarks import Landmarks
from bottle_learned import AsyncBottleFinder
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH
from realtime import FramePacket, compose_preview


def observed(frame, blocked=None):
    blocked = np.zeros(frame.shape[:2], np.uint8) if blocked is None else blocked
    seed = Landmarks(CAP, BASE, BODY_WIDTH, .9, 'learned_bottle_box')
    first = observe_endpoints(frame, seed, blocked)
    if first is None:
        raise AssertionError('Synthetic cap and bottom must be visible')
    return observe_endpoints(frame, first, blocked)


class GripTransitionTests(unittest.TestCase):
    def test_neck_to_body_grip_recovers_visible_endpoints_and_preserves_baseline(self):
        before, _ = axis_frame()
        cv2.rectangle(before, (245, 88), (395, 130), (70, 125, 180), -1)
        neck = np.zeros(before.shape[:2], np.uint8)
        neck[88:131, 245:396] = 255
        seed = observed(before, neck)
        hand = [(320, 250)] * 21
        with patch('bottle_axis.hand_mask', return_value=neck), \
                patch('bottle_axis.body_occlusion_mask', return_value=neck):
            tracker = AxisBottleTracker(before, seed.cap, seed.base, hand, BODY_WIDTH, seed.source)
            self.assertTrue(tracker.set_baseline([tracker.last] * 4))
        baseline = tracker.baseline_left.copy(), tracker.baseline_right.copy()
        after, _ = axis_frame()
        cv2.rectangle(after, (225, 175), (390, 325), (70, 125, 180), -1)
        body = np.zeros(before.shape[:2], np.uint8)
        body[172:329, 222:394] = 255
        tracker.endpoint_tracker.allow_geometry = False
        with patch('bottle_axis.hand_mask', return_value=body), \
                patch('bottle_axis.body_occlusion_mask', return_value=body), \
                patch('bottle_axis.cv2.calcOpticalFlowPyrLK', return_value=(None, None, None)), \
                patch.object(tracker.endpoint_tracker, '_template_fallback', return_value=None):
            measured = tracker.measure(after, hand, 1.)
        self.assertIsNotNone(measured.axis_points, measured.reason)
        self.assertEqual(tracker.endpoint_tracker.local_recoveries, 1)
        np.testing.assert_allclose(tracker.pose.cap, seed.cap, atol=3)
        np.testing.assert_allclose(tracker.pose.base, seed.base, atol=4)
        np.testing.assert_allclose(tracker.baseline_left, baseline[0], equal_nan=True)
        np.testing.assert_allclose(tracker.baseline_right, baseline[1], equal_nan=True)
        self.assertTrue(any(v is None for v in measured.profile_left))
        self.assertIsNone(measured.metrics['visible_area_reduction_pct'])

    def test_reobserved_ai_candidate_refreshes_motion_template_not_deformation_baseline(self):
        first, _ = axis_frame()
        seed = observed(first)
        tracker = AxisBottleTracker(first, seed.cap, seed.base, [], BODY_WIDTH, seed.source)
        self.assertTrue(tracker.set_baseline([tracker.last] * 4))
        baseline = tracker.baseline_left.copy()
        current, _ = axis_frame(shift=(110., 0.))
        proposal = Landmarks(seed.cap + (110, 0), seed.base + (110, 0), BODY_WIDTH, .9, 'learned_bottle_box')
        candidate = observe_endpoints(current, proposal, np.zeros(first.shape[:2], np.uint8),
                                      tracker.endpoint_tracker.cap_signature)
        self.assertIsNotNone(candidate)
        with patch.object(tracker.endpoint_tracker, '_verify', return_value=0.):
            adopted = tracker.endpoint_tracker.adopt_geometry(current, np.zeros(first.shape[:2], np.uint8), candidate)
        self.assertIsNotNone(adopted)
        result = tracker.measure(current, [], 1.)
        self.assertIsNotNone(result.axis_points)
        np.testing.assert_allclose(tracker.baseline_left, baseline, equal_nan=True)
        np.testing.assert_array_equal(tracker.endpoint_tracker.previous, cv2.cvtColor(current, cv2.COLOR_BGR2GRAY))

    @staticmethod
    def async_result(source, seed):
        result = {'frame_id': 1, 'captured_at': 1., 'frame': source, 'hand': [], 'generation': 0,
                  'candidate': seed, 'detection': None, 'status': 'found', 'model_name': 'test'}
        finder = AsyncBottleFinder.__new__(AsyncBottleFinder)
        finder.worker = SimpleNamespace(error=None, submit=lambda _: None, read_latest=lambda: result)
        finder._request_id = finder._seen_id = -1
        finder._generation = 0
        return finder

    def test_async_bridge_uses_fresh_pixels_when_optical_flow_cannot_transfer_result(self):
        source, _ = axis_frame()
        seed = observed(source)
        current, _ = axis_frame(shift=(10., 0.))
        finder = self.async_result(source, seed)
        with patch('bottle_learned.EndpointTracker.track', return_value=(None, 0., 'no flow')) as old_bridge:
            result = finder.update(current, [], 1.15)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.endpoint_quality)
        np.testing.assert_allclose(result.cap, seed.cap + (10, 0), atol=3)
        np.testing.assert_allclose(result.base, seed.base + (10, 0), atol=4)
        old_bridge.assert_not_called()

    def test_async_bridge_does_not_accept_missing_base_just_because_ai_found_a_box(self):
        source, _ = axis_frame()
        seed = observed(source)
        current = source.copy()
        current[365:] = 165
        finder = self.async_result(source, seed)
        with patch('bottle_learned.EndpointTracker.track', return_value=(None, 0., 'base missing')):
            self.assertIsNone(finder.update(current, [], 1.15))

    def test_hidden_cap_or_hidden_base_has_no_geometric_observation(self):
        frame, _ = axis_frame()
        seed = observed(frame)
        identity = cap_identity(frame, seed.cap, BODY_WIDTH)
        for lo, hi in ((20, 110), (365, 450)):
            blocked = np.zeros(frame.shape[:2], np.uint8)
            blocked[lo:hi] = 255
            self.assertIsNone(observe_endpoints(frame, seed, blocked, identity))

    def test_cap_and_background_stripe_without_body_sides_are_not_a_bottle(self):
        frame, _ = axis_frame()
        seed = observed(frame)
        current = frame.copy()
        current[100:] = 165
        cv2.line(current, (220, 410), (420, 410), (80, 80, 80), 3)
        self.assertIsNone(observe_endpoints(current, seed, np.zeros(frame.shape[:2], np.uint8)))

    def test_recovery_rejects_a_different_cap_even_with_valid_endpoint_geometry(self):
        frame, _ = axis_frame()
        seed = observed(frame)
        tracker = AxisBottleTracker(frame, seed.cap, seed.base, [], BODY_WIDTH, seed.source)
        current = frame.copy()
        cv2.rectangle(current, (285, 48), (355, 82), (220, 70, 30), -1)
        other = observe_endpoints(current, seed, np.zeros(frame.shape[:2], np.uint8))
        self.assertIsNotNone(other)
        self.assertIsNone(tracker.endpoint_tracker.adopt_geometry(current, np.zeros(frame.shape[:2], np.uint8), other))

    def test_preview_compositing_preserves_exact_nonblack_pixels_and_live_background(self):
        frame = np.full((480, 640, 3), (17, 23, 41), np.uint8)
        canvas = np.zeros((650, 1000, 3), np.uint8)
        # Include tiny single-channel antialiasing values, which a grayscale
        # mask can round to zero, and a parent image with a larger row stride.
        values = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (255, 205, 70), (0, 0, 0))
        for i, value in enumerate(values):
            canvas[100, 100 + i] = value
        result = {'canvas': canvas, 'captured_at': 1., 'processing_ms': 50}
        output = compose_preview(FramePacket(1, 1.1, frame), result, 1.1)
        for i, value in enumerate(values):
            expected = value if any(value) else (17, 23, 41)
            np.testing.assert_array_equal(output[100, 100 + i], expected)


if __name__ == '__main__':
    unittest.main()
