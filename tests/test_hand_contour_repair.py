"""False palm/background contours and real-time joint-baseline failures."""

import unittest

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_detector import Detection
from bottle_endpoints import observe_endpoints
from bottle_landmarks import Landmarks
from bottle_joint_estimate import JointContourEstimator
from bottle_learned import detection_landmarks
from bottle_occlusion import body_occlusion_mask
from bottle_outline import hand_mask
from test_axis import axis_frame, CAP, BASE, BODY_WIDTH
from test_joint_estimate import fixture, POSE, STATION_ROWS


class HandContourRepairTests(unittest.TestCase):
    def test_cropped_bottom_seed_does_not_accept_an_interior_rib_as_the_foot(self):
        frame, _ = axis_frame()
        for y in range(340, 401, 9):
            cv2.line(frame, (260, y), (380, y), (105, 105, 105), 2)
        seed = Landmarks(CAP, BASE - (0, 40), BODY_WIDTH, .9, 'learned_bottle_box')
        result = observe_endpoints(frame, seed, np.zeros(frame.shape[:2], np.uint8),
                                   region=(250, 48, 140, 345))
        self.assertIsNotNone(result)
        # The fixture's bottom ellipse is centered at BASE and extends 10px
        # downward. Its terminating rim, rather than an interior rib, is valid.
        self.assertGreaterEqual(result.base[1], BASE[1] - 8)
        self.assertLessEqual(result.base[1], BASE[1] + 12)

    def test_wrap_grip_palm_is_excluded_beyond_bones_without_hiding_clear_foot(self):
        hand = [(445, 342), (400, 330), (370, 320), (346, 310), (330, 300),
                (405, 250), (355, 238), (325, 235), (300, 235),
                (410, 270), (355, 265), (323, 265), (300, 265),
                (414, 292), (355, 292), (325, 292), (300, 292),
                (420, 316), (360, 320), (333, 320), (312, 320)]
        frame = np.full((480, 640, 3), 165, np.uint8)
        geometry = hand_mask(frame.shape, hand)
        frame[geometry > 0] = (80, 130, 190)
        cv2.rectangle(frame, (420, 260), (500, 325), (80, 130, 190), -1)
        cv2.rectangle(frame, (50, 50), (100, 100), (80, 130, 190), -1)
        cv2.line(frame, (465, 375), (465, 415), (130, 130, 130), 2)
        mask = body_occlusion_mask(frame, hand)
        self.assertEqual(np.count_nonzero(geometry[275:316, 480:496]), 0)
        self.assertGreater(np.mean(mask[275:316, 480:496] > 0), .9)
        self.assertFalse(np.any(mask[375:416, 460:470]))
        self.assertFalse(np.any(mask[50:101, 50:101]))

    def test_wide_box_and_continuing_background_lines_do_not_define_body_width(self):
        frame, _ = axis_frame()
        cv2.line(frame, (220, 110), (220, 465), (25, 25, 25), 3)
        cv2.line(frame, (430, 110), (430, 465), (25, 25, 25), 3)
        seed = detection_landmarks(frame, Detection((200, 48, 240, 371), .9), [])
        self.assertIsNotNone(seed.endpoint_quality)
        self.assertAlmostEqual(seed.body_width, 130., delta=10.)
        tracker = AxisBottleTracker(frame, seed.cap, seed.base, [], seed.body_width, seed.source)
        self.assertAlmostEqual(tracker.last.width_px, 130., delta=7.)
        canvas = np.zeros_like(frame)
        tracker.draw(canvas)
        blue = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] > 150) & (canvas[:, :, 2] < 120)
        self.assertFalse(np.any(blue[150:350, 210:231]))
        self.assertFalse(np.any(blue[150:350, 425:436]))

    def test_forearm_outside_bottle_cannot_hide_all_upper_lower_anchors(self):
        hand, mask, segments = fixture()
        mask[75:350, 260:295] = 255  # Arm extends outside the bottle corridor.
        mask[245:275, 230:295] = 255
        estimator = JointContourEstimator(STATION_ROWS, 1., body_width=120.)
        estimator.request(0.)
        for stamp in (3., 3.2, 3.4, 3.6, 3.8):
            estimator.update(POSE, hand, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertTrue(estimator.segments)

    def test_baseline_can_wait_for_edges_instead_of_expiring_after_one_second(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        missing_upper_right = [segments[i] for i in (0, 1, 3)]
        for stamp in (3.1, 4., 5.):
            estimator.update(POSE, hand, missing_upper_right, mask, stamp)
        self.assertIn('RIGHT', estimator.status)
        self.assertIn('ABOVE', estimator.status)
        self.assertFalse(estimator.ready)
        for stamp in (5.2, 5.4, 5.6, 5.8):
            estimator.update(POSE, hand, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 0.)

    def test_missing_side_stays_pending_and_recovers_without_another_h_press(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for stamp in (3.1, 3.3, 8.1):
            estimator.update(POSE, hand, [segments[i] for i in (0, 2, 3)], mask, stamp)
        self.assertFalse(estimator.ready)
        self.assertTrue(estimator.calibrating)
        self.assertIn('LEFT', estimator.status)
        self.assertIn('BELOW', estimator.status)
        self.assertIsNone(estimator.metrics['joint_estimated_width_px'])
        for stamp in (10., 10.2, 10.4, 10.6):
            estimator.update(POSE, hand, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertTrue(estimator.segments)

    def test_axis_loss_at_old_timeout_does_not_cancel_the_requested_estimate(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        estimator.missing(9., 'Estimate paused: bottle axis lost')
        self.assertTrue(estimator.calibrating)
        self.assertIn('bottle axis lost', estimator.status)
        self.assertIsNone(estimator.metrics['joint_estimated_width_px'])
        for stamp in (12., 12.2, 12.4, 12.6):
            estimator.update(POSE, hand, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertTrue(estimator.segments)

    def test_one_bad_hand_frame_does_not_erase_a_coherent_rest_group(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for stamp in (3., 3.2, 3.3, 3.4, 3.6):
            current = hand + ((25, 0) if stamp == 3.3 else (0, 0))
            estimator.update(POSE, current, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 0.)

    def test_rest_jitter_is_compared_to_the_window_median_instead_of_one_extreme_frame(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for tick, stamp in enumerate((3., 3.2, 3.4, 3.6)):
            estimator.update(POSE, hand + (4 if tick % 2 else -4, 0), segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)
        self.assertGreater(len(estimator.segments), 0)

    def test_one_jittering_fingertip_does_not_reset_a_stable_rest_palm(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for tick, stamp in enumerate((3., 3.2, 3.4, 3.6, 3.8)):
            current = hand.copy()
            current[8, 0] += 14 * (tick % 2)
            estimator.update(POSE, current, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)

    def test_old_samples_are_not_combined_with_a_later_grip_after_a_long_gap(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for stamp in (3.1, 3.3, 3.5):
            estimator.update(POSE, hand, segments, mask, stamp)
        estimator.update(POSE, hand, segments[:1], mask, 4.)
        estimator.update(POSE, hand, segments, mask, 6.)
        self.assertFalse(estimator.ready)
        for stamp in (6.2, 6.4, 6.6):
            estimator.update(POSE, hand, segments, mask, stamp)
        self.assertTrue(estimator.ready, estimator.status)


if __name__ == '__main__':
    unittest.main()
