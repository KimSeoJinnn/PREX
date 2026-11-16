import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_occlusion import body_occlusion_mask
from bottle_outline import hand_mask
from bottle_landmarks import Landmarks
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH


class BodyOcclusionTests(unittest.TestCase):
    def test_visible_right_edge_remains_drawn_when_left_edge_is_hidden(self):
        tracker = self.tracker()
        frame, _ = axis_frame()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[:, 230:275] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked):
            measured = tracker.measure(frame, [], 1.)
        self.assertIsNotNone(measured.axis_points)
        self.assertIsNone(measured.width_px)
        self.assertTrue(all(value is None for value in measured.profile_left))
        self.assertGreater(np.count_nonzero(np.isfinite(tracker.visible_right)), 25)
        self.assertEqual(np.count_nonzero(np.isfinite(tracker.visible_left)), 0)
        self.assertIsNone(measured.metrics['median_width_reduction_pct'])
        self.assertGreater(measured.metrics['right_edge_coverage'], .8)
        canvas = np.zeros_like(frame)
        tracker.draw(canvas)
        self.assertTrue(np.any(canvas[175:350, 380:390]))
        self.assertFalse(np.any(canvas[:, 230:275]))

    def test_neck_and_base_can_be_displayed_without_creating_body_measurements(self):
        frame, _ = axis_frame()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[150:380] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked):
            tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        self.assertIsNone(tracker.last.width_px)
        self.assertTrue(all(value is None for value in tracker.last.profile_left))
        self.assertGreater(len(tracker.outline_segments), 0)
        canvas = np.zeros_like(frame)
        tracker.draw(canvas)
        self.assertTrue(np.any(canvas[380:405, 250:265]))
        self.assertFalse(tracker.set_baseline([tracker.last] * 4))

    def test_large_relocation_can_recover_verified_same_bottle_without_resetting_baseline(self):
        tracker = self.tracker()
        reference = tracker.baseline_left.copy()
        frame, pose = axis_frame(shift=(110., 0.))
        candidate = Landmarks(pose.cap, pose.base, BODY_WIDTH, .9, 'learned_bottle_box')
        result = tracker.endpoint_tracker.adopt_geometry(frame, np.zeros(frame.shape[:2], np.uint8), candidate)
        self.assertIsNotNone(result)
        measured = tracker.measure(frame, [], 1.)
        self.assertIsNotNone(measured.width_px)
        self.assertAlmostEqual(measured.metrics['median_width_reduction_pct'], 0, delta=1)
        np.testing.assert_allclose(tracker.baseline_left, reference, equal_nan=True)

    def test_large_relocation_rejects_incompatible_cap_appearance(self):
        tracker = self.tracker()
        frame, pose = axis_frame(shift=(110., 0.))
        cv2.rectangle(frame, (390, 35), (470, 90), (120, 120, 120), -1)
        candidate = Landmarks(pose.cap, pose.base, BODY_WIDTH, .9, 'learned_bottle_box')
        self.assertIsNone(tracker.endpoint_tracker.adopt_geometry(frame, np.zeros(frame.shape[:2], np.uint8), candidate))

    def test_skin_rim_outside_landmark_strokes_is_excluded_without_selecting_remote_color(self):
        frame = np.full((480, 640, 3), 165, np.uint8)
        hand = [(445, 342), (400, 330), (370, 320), (346, 310), (330, 300),
                (405, 250), (355, 238), (325, 235), (300, 235),
                (410, 270), (355, 265), (323, 265), (300, 265),
                (414, 292), (355, 292), (325, 292), (300, 292),
                (420, 316), (360, 320), (333, 320), (312, 320)]
        truth = np.zeros(frame.shape[:2], np.uint8)
        points = np.asarray(hand, np.int32)
        cv2.fillConvexPoly(truth, cv2.convexHull(points[[0, 1, 2, 5, 9, 13, 17]]), 255)
        for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            for a, b in zip(chain, chain[1:]):
                cv2.line(truth, hand[a], hand[b], 255, 36)
            for index in chain:
                cv2.circle(truth, hand[index], 18, 255, -1)
        frame[truth > 0] = (80, 130, 190)
        cv2.rectangle(frame, (50, 50), (100, 100), (80, 130, 190), -1)
        geometry = hand_mask(frame.shape, hand)
        mask = body_occlusion_mask(frame, hand)
        uncovered_skin = (truth > 0) & (geometry == 0)
        self.assertGreater(np.count_nonzero(uncovered_skin), 500)
        self.assertGreater(np.mean(mask[uncovered_skin] > 0), .90)
        self.assertEqual(np.count_nonzero(mask[50:101, 50:101]), 0)

    def test_missing_landmarks_do_not_classify_entire_image_by_color(self):
        frame = np.full((480, 640, 3), (80, 130, 190), np.uint8)
        self.assertFalse(np.any(body_occlusion_mask(frame, [])))
        neutral = np.full_like(frame, 165)
        mask = body_occlusion_mask(neutral, [(500, 250)] * 21)
        self.assertLess(np.count_nonzero(mask), 1000)

    @staticmethod
    def tracker():
        frame, _ = axis_frame()
        tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
        assert tracker.set_baseline([tracker.measure(frame, [], i / 30) for i in range(4)])
        return tracker

    def test_hidden_edge_cannot_escape_to_stronger_hand_or_background_edge(self):
        tracker = self.tracker()
        frame, pose = axis_frame()
        cv2.line(frame, (235, 205), (235, 305), (30, 30, 30), 3)
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[205:305, 247:270] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked), patch.object(
                tracker.endpoint_tracker, 'track', return_value=(pose, 1., 'Axis OK')):
            measured = tracker.measure(frame, [], 1.)
        image_y = CAP[1] + tracker.positions * (BASE[1] - CAP[1])
        hidden = (image_y >= 215) & (image_y <= 295)
        self.assertTrue(all(v is None for v in np.asarray(measured.profile_left, object)[hidden]))
        self.assertGreater(measured.metrics['hand_occluded_profiles'], 8)
        self.assertAlmostEqual(measured.metrics['median_width_reduction_pct'], 0, delta=1)

    def test_body_and_baseline_overlay_do_not_cross_occluding_hand(self):
        tracker = self.tracker()
        frame, pose = axis_frame()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[220:290, 290:345] = 255  # Both outer edges remain visible.
        with patch('bottle_axis.hand_mask', return_value=blocked), patch.object(
                tracker.endpoint_tracker, 'track', return_value=(pose, 1., 'Axis OK')):
            measured = tracker.measure(frame, [], 1.)
        self.assertIsNotNone(measured.width_px)
        canvas = np.zeros_like(frame)
        tracker.draw(canvas)
        # The yellow cap-base axis is a pose annotation. Body and baseline
        # lines must leave hand pixels untouched on either side of the axis.
        check = blocked > 0
        check[:, 316:325] = False
        self.assertFalse(np.any(canvas[check]))
        self.assertTrue(np.any(canvas[180:310, 250:260]))

    def test_lost_hand_pauses_body_measurement_and_recovers_with_same_baseline(self):
        tracker = self.tracker()
        frame, _ = axis_frame()
        hand = [(500, 250)] * 21
        reference = tracker.baseline_left.copy()
        self.assertIsNotNone(tracker.measure(frame, hand, 1.).width_px)
        missing = tracker.measure(frame, [], 2.)
        self.assertIsNotNone(missing.axis_points)
        self.assertIsNone(missing.width_px)
        self.assertIsNone(missing.metrics['median_width_reduction_pct'])
        self.assertFalse(missing.edges)
        self.assertIn('Hand lost', missing.reason)
        self.assertIsNotNone(tracker.measure(frame, hand, 3.).width_px)
        np.testing.assert_allclose(tracker.baseline_left, reference, equal_nan=True)

    def test_initial_hidden_body_does_not_become_a_baseline(self):
        frame, _ = axis_frame()
        blocked = np.zeros(frame.shape[:2], np.uint8)
        blocked[150:380, 230:410] = 255
        with patch('bottle_axis.hand_mask', return_value=blocked):
            tracker = AxisBottleTracker(frame, CAP, BASE, [], BODY_WIDTH)
            samples = [tracker.measure(frame, [], i / 30) for i in range(4)]
        self.assertIsNotNone(tracker.last.axis_points)
        self.assertIsNone(tracker.last.width_px)
        self.assertFalse(tracker.set_baseline(samples))
        self.assertTrue(all(v is None for v in tracker.last.profile_left))


if __name__ == '__main__':
    unittest.main()
