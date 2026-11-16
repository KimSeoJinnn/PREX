"""Image-supported continuity must move with the hand and expire honestly."""

import unittest

import cv2
import numpy as np

from bottle_hand_tracking import GripHandTracker
from test_joint_estimate import fixture


class GripTrackingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(36)
        self.frame = rng.integers(45, 210, (384, 320, 3), dtype=np.uint8)
        self.hand, _, _ = fixture()
        self.tracker = GripHandTracker()

    def moved(self, dx, dy):
        return cv2.warpAffine(self.frame, np.float32([[1, 0, dx], [0, 1, dy]]), (320, 384))

    def test_short_missing_detection_moves_with_the_image_instead_of_freezing(self):
        self.tracker.update(self.frame, self.hand, 'Right', 0.)
        result = self.tracker.update(self.moved(3, -2), [], None, .1)
        self.assertTrue(result.display_only)
        self.assertAlmostEqual(result.age_ms, 100.)
        np.testing.assert_allclose(result.points, self.hand + (3, -2), atol=.25)
        result = self.tracker.update(self.moved(6, -4), [], None, .2)
        np.testing.assert_allclose(result.points, self.hand + (6, -4), atol=.4)
        self.assertAlmostEqual(result.age_ms, 200.)

    def test_flow_cannot_renew_itself_forever_without_a_hand_detection(self):
        self.tracker.update(self.frame, self.hand, 'Right', 0.)
        for stamp in (.1, .2, .3, .4):
            self.assertTrue(self.tracker.update(self.frame, [], None, stamp).display_only)
        self.assertFalse(self.tracker.update(self.frame, [], None, .46).points)
        self.assertEqual(self.tracker.last.source, 'missing')

    def test_detector_jump_is_rejected_and_label_flip_alone_does_not_change_identity(self):
        self.tracker.update(self.frame, self.hand, 'Right', 0.)
        result = self.tracker.update(self.moved(2, 0), self.hand + (60, 0), 'Left', .1)
        self.assertTrue(result.display_only)
        np.testing.assert_allclose(result.points, self.hand + (2, 0), atol=.25)
        result = self.tracker.update(self.moved(4, 0), self.hand + (4, 0), 'Left', .2)
        self.assertEqual(result.source, 'landmarks')
        self.assertEqual(result.side, 'Right')
        np.testing.assert_allclose(result.points, self.hand + (4, 0), atol=.25)

    def test_no_image_support_or_a_scene_cut_clears_the_guide(self):
        flat = np.full_like(self.frame, 165)
        self.tracker.update(flat, self.hand, 'Right', 0.)
        self.assertFalse(self.tracker.update(flat, [], None, .1).points)
        self.tracker.update(self.frame, self.hand, 'Right', 1.)
        self.assertFalse(self.tracker.update(np.zeros_like(self.frame), [], None, 1.1).points)

    def test_repeated_timestamp_does_not_extend_the_detection_deadline(self):
        self.tracker.update(self.frame, self.hand, 'Right', 0.)
        self.tracker.update(self.frame, [], None, .1)
        for _ in range(10):
            self.tracker.update(self.frame, [], None, .1)
        self.assertFalse(self.tracker.update(self.frame, [], None, .46).points)


if __name__ == '__main__':
    unittest.main()
