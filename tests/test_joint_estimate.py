"""A calibrated joint proxy must stay separate from observed bottle widths."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bottle_axis import AxisBottleTracker, BottlePose
from bottle_joint_estimate import JointContourEstimator, ESTIMATE_COLOR
from main import Hand, RealtimeSession
from realtime import FramePacket, save_debug_snapshot


POSE = BottlePose(np.array([160., 32.]), np.array([160., 352.]))
STATION_ROWS = 32 + np.linspace(.27, .87, 31) * 320


def fixture(compression=0.):
    hand = np.array([(220, 300), (210, 280), (220, 200), (220, 180), (120, 180),
                     (220, 210), (190, 200), (150, 200), (120, 200),
                     (220, 240), (190, 230), (150, 230), (120, 230),
                     (220, 270), (190, 260), (150, 260), (120, 260),
                     (220, 290), (190, 285), (150, 290), (120, 300)], float)
    hand[:, 0] = 170 + (hand[:, 0] - 170) * (1 - compression)
    mask = np.zeros((384, 320), np.uint8)
    mask[165:316, 95:246] = 255
    segments = [np.column_stack((np.full(5, x), ys)).astype(float)
                for x in (110, 230) for ys in ((135, 140, 145, 150, 155), (325, 330, 335, 340, 345))]
    return hand, mask, segments


def calibrated():
    estimator = JointContourEstimator(STATION_ROWS, 1.)
    hand, mask, segments = fixture()
    estimator.request(0.)
    for stamp in (3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4):
        estimator.update(POSE, hand, segments, mask, stamp)
    return estimator


class JointEstimateTests(unittest.TestCase):
    def test_hand_identity_flips_are_not_combined_or_reported_as_compression(self):
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        hand, mask, segments = fixture()
        estimator.request(0.)
        for stamp, side in ((3., 'Right'), (3.2, 'Right'), (3.4, 'Left'), (3.6, 'Left')):
            estimator.update(POSE, hand, segments, mask, stamp, hand_side=side)
        self.assertFalse(estimator.ready)
        for stamp in (3.8, 4.):
            estimator.update(POSE, hand, segments, mask, stamp, hand_side='Right')
        self.assertTrue(estimator.ready, estimator.status)
        estimator.update(POSE, hand, segments, mask, 4.1, hand_side='Left')
        self.assertFalse(estimator.segments)
        self.assertIsNone(estimator.metrics['joint_estimated_width_px'])
        self.assertIn('identity changed', estimator.status)
        estimator.update(POSE, hand, segments, mask, 4.2, hand_side='Right')
        self.assertTrue(estimator.segments)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 0.)

    def test_rest_is_zero_and_skeleton_narrowing_changes_proxy_by_known_amount(self):
        estimator = calibrated()
        self.assertTrue(estimator.ready, estimator.status)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_px'], 120., places=4)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 0., places=4)
        hand, mask, segments = fixture(.2)
        estimator.update(POSE, hand, segments, mask, 5.)
        # Hull width 100 -> 80, preserving the two 10px rest offsets:
        # inferred bottle width 120 -> 100, i.e. a 16 2/3 percent reduction.
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_px'], 100., places=4)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 100 / 6, places=4)
        self.assertGreater(len(estimator.segments), 0)
        json.dumps(estimator.context(), allow_nan=False)

    def test_image_similarity_motion_does_not_create_proxy_compression(self):
        estimator = calibrated()
        hand, mask, segments = fixture()
        matrix = cv2.getRotationMatrix2D((160., 192.), 9., .94)
        matrix[:, 2] += (5., 3.)
        transform = lambda p: cv2.transform(np.asarray(p, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)
        endpoints = transform([POSE.cap, POSE.base])
        pose = BottlePose(*endpoints)
        mask = cv2.warpAffine(mask, matrix, (320, 384), flags=cv2.INTER_NEAREST)
        estimator.update(pose, transform(hand), [transform(s) for s in segments], mask, 5.)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 0., delta=.1)

    def test_missing_hand_clears_values_but_hidden_anchors_can_use_the_reference(self):
        estimator = calibrated()
        hand, mask, segments = fixture(.2)
        estimator.update(POSE, hand, segments, mask, 5.)
        self.assertIsNotNone(estimator.metrics['joint_estimated_width_px'])
        estimator.missing(5.1, 'Hand lost')
        self.assertTrue(estimator.ready)
        self.assertTrue(np.isnan(estimator.profile_left).all())
        self.assertFalse(estimator.segments)
        self.assertIsNone(estimator.metrics['joint_estimated_width_px'])
        estimator.update(POSE, hand, segments[:1], mask, 5.2)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 100 / 6, places=4)
        self.assertEqual(estimator.anchor_source, 'calibrated_reference')
        self.assertTrue(estimator.segments)
        estimator.update(POSE, hand, segments, mask, 5.3)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 100 / 6, places=4)

    def test_no_runtime_anchor_is_needed_and_new_pose_projects_the_reference(self):
        estimator = calibrated()
        hand, mask, _ = fixture(.2)
        estimator.update(POSE, hand, [], mask, 5.)
        self.assertTrue(estimator.segments)
        self.assertAlmostEqual(estimator.metrics['joint_estimated_width_reduction_pct'], 100 / 6, places=4)
        first = [s.copy() for s in estimator.segments]
        translated = BottlePose(POSE.cap + (7, 4), POSE.base + (7, 4))
        shifted_mask = cv2.warpAffine(mask, np.float32([[1, 0, 7], [0, 1, 4]]), (320, 384))
        estimator.update(translated, hand + (7, 4), [], shifted_mask, 5.1)
        self.assertEqual(len(first), len(estimator.segments))
        for a, b in zip(first, estimator.segments):
            np.testing.assert_allclose(b, a + (7, 4), atol=.001)

    def test_flow_only_guide_draws_without_widths_profiles_or_calibration_samples(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for stamp in (3., 3.2, 3.4, 3.6):
            estimator.update(POSE, hand, segments, mask, stamp, display_only=True)
        self.assertFalse(estimator.ready)
        self.assertEqual(estimator.samples, [])
        estimator = calibrated()
        estimator.update(POSE, hand, [], mask, 5., display_only=True)
        self.assertTrue(estimator.segments)
        self.assertTrue(estimator.display_only)
        self.assertTrue(all(v is None for v in estimator.metrics.values()))
        self.assertTrue(np.isnan(estimator.profile_left).all())
        self.assertTrue(np.isnan(estimator.profile_right).all())

    def test_duplicate_frames_and_motion_cannot_finish_rest_calibration(self):
        hand, mask, segments = fixture()
        estimator = JointContourEstimator(STATION_ROWS, 1.)
        estimator.request(0.)
        for _ in range(10):
            estimator.update(POSE, hand, segments, mask, 3.1)
        estimator.update(POSE, hand, segments, mask, 4.4)
        self.assertFalse(estimator.ready)
        estimator.request(10.)
        for tick in range(6):
            estimator.update(POSE, hand + (12 * (tick % 2), 0), segments, mask, 13 + tick * .2)
        estimator.update(POSE, hand, segments, mask, 14.4)
        self.assertFalse(estimator.ready)

    def test_relocated_or_rotated_grip_has_no_stale_proxy(self):
        estimator = calibrated()
        hand, mask, segments = fixture()
        for changed in (hand + (60, 0), np.column_stack((hand[:, 0], 480 - hand[:, 1]))):
            estimator.update(POSE, changed, segments, mask, 5.)
            self.assertIsNone(estimator.metrics['joint_estimated_width_px'])
            self.assertFalse(estimator.segments)

    def test_overlay_has_both_proxy_sides_and_uses_distinct_color(self):
        estimator = calibrated()
        canvas = np.zeros((384, 320, 3), np.uint8)
        estimator.draw(canvas)
        color = np.all(canvas == ESTIMATE_COLOR, axis=2)
        self.assertTrue(np.any(color[190:290, 108:113]))
        self.assertTrue(np.any(color[190:290, 228:233]))
        self.assertFalse(np.any(canvas[:165]))
        # Dashes leave gaps rather than painting a continuous measured edge.
        self.assertTrue(np.any(~np.any(canvas[190:290, 108:113] > 0, axis=(1, 2))))

    def test_real_visible_edge_is_not_overpainted_as_inferred(self):
        estimator = calibrated()
        hand, mask, segments = fixture()
        segments.append(np.column_stack((np.full(121, 110.), np.arange(180, 301))))
        estimator.update(POSE, hand, segments, mask, 5.)
        self.assertTrue(estimator.segments)
        self.assertTrue(all(np.mean(segment[:, 0]) > 200 for segment in estimator.segments))


class JointEstimateSessionTests(unittest.TestCase):
    def test_short_hand_gap_keeps_a_moving_guide_but_records_no_readings(self):
        hand, mask, segments = fixture()
        blank = np.full((384, 320, 3), 165, np.uint8)
        rng = np.random.default_rng(61)
        frame = rng.integers(45, 210, blank.shape, dtype=np.uint8)
        model = SimpleNamespace(close=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            session = RealtimeSession(model, None, Path(directory), {}, 'test')
            band = AxisBottleTracker(blank, POSE.cap, POSE.base, hand.tolist(), 120.)
            session.band = band

            def extract(*_):
                band.outline_segments = segments
                empty = np.full(31, np.nan)
                return empty.copy(), empty.copy(), np.ones(31), np.ones(31)

            try:
                with patch('main.observe_hand', return_value=Hand('Right', hand.astype(int).tolist(), np.full(14, 150.))) as detect, \
                        patch.object(band.endpoint_tracker, 'track', return_value=(POSE, 1., 'Axis OK')), \
                        patch('bottle_axis.hand_mask', return_value=mask), \
                        patch('bottle_axis.body_occlusion_mask', return_value=mask), \
                        patch.object(band, '_extract', side_effect=extract):
                    session.process(FramePacket(0, 100., frame), ['h'])
                    for tick, stamp in enumerate((3., 3.2, 3.4, 3.6), 1):
                        result = session.process(FramePacket(tick, 100 + stamp, frame), [])
                    self.assertTrue(result['joint_estimate_ready'], result['joint_estimate_status'])
                    session.process(FramePacket(5, 103.8, frame), ['r'])
                    detect.return_value = None
                    moved = cv2.warpAffine(frame, np.float32([[1, 0, 3], [0, 1, 0]]), (320, 384))
                    gap = session.process(FramePacket(6, 104., moved), [])
                    self.assertFalse(gap['hand_detected'])
                    self.assertTrue(gap['joint_display_only'])
                    self.assertGreater(gap['joint_estimate_segments'], 0)
                    self.assertIsNone(gap['joint_estimated_width_px'])
                    self.assertIsNone(band.last.width_px)
                    canvas = np.zeros_like(frame)
                    band.draw(canvas)
                    self.assertTrue(np.any(np.all(canvas == ESTIMATE_COLOR, axis=2)))
                    expired = session.process(FramePacket(7, 104.6, moved), [])
                    self.assertEqual(expired['joint_estimate_segments'], 0)
                    session.process(FramePacket(8, 104.8, moved), ['r'])
                with next(Path(directory).glob('*.csv')).open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                self.assertNotEqual(rows[0]['joint_estimated_width_px'], '')
                self.assertEqual(rows[1]['joint_hand_source'], 'optical_flow')
                self.assertEqual(rows[1]['joint_display_only'], 'True')
                self.assertEqual(rows[2]['joint_hand_source'], 'missing')
                for row in rows[1:]:
                    self.assertEqual(row['joint_estimated_width_px'], '')
                    self.assertEqual(row['bottle_width_px'], '')
                    self.assertEqual(row['thumb_mcp_deg'], '')
                    self.assertTrue(all(v is None for v in json.loads(row['joint_estimated_left_canonical_px'])))
            finally:
                session.close()

    def test_h_before_bottle_acquisition_is_remembered_until_a_bottle_is_found(self):
        hand, mask, segments = fixture()
        frame = np.full((384, 320, 3), 165, np.uint8)
        model = SimpleNamespace(close=lambda: None)
        session = RealtimeSession(model, None, Path('.'), {}, 'test')
        band = AxisBottleTracker(frame, POSE.cap, POSE.base, [], 120.)
        session.finder = SimpleNamespace(acquire=lambda *_: None, status='looking')

        def extract(*_):
            band.outline_segments = segments
            empty = np.full(31, np.nan)
            return empty.copy(), empty.copy(), np.ones(31), np.ones(31)

        try:
            with patch('main.observe_hand', return_value=Hand('Right', hand.astype(int).tolist(), np.full(14, 150.))), \
                    patch.object(band.endpoint_tracker, 'track', return_value=(POSE, 1., 'Axis OK')), \
                    patch('bottle_axis.hand_mask', return_value=mask), \
                    patch('bottle_axis.body_occlusion_mask', return_value=mask), \
                    patch.object(band, '_extract', side_effect=extract):
                waiting = session.process(FramePacket(0, 100., frame), ['h'])
                self.assertTrue(waiting['joint_estimate_requested'])
                self.assertIn('waiting', waiting['status'])
                session.finder.acquire = lambda *_: band
                found = session.process(FramePacket(1, 102., frame), [])
                self.assertFalse(found['joint_estimate_requested'])
                self.assertTrue(found['joint_estimate_calibrating'])
                for tick, stamp in enumerate((105.1, 105.3, 105.5, 105.7), 2):
                    result = session.process(FramePacket(tick, stamp, frame), [])
                self.assertTrue(result['joint_estimate_ready'], result['joint_estimate_status'])
                self.assertGreater(result['joint_estimate_segments'], 0)
        finally:
            session.close()

    def test_proxy_only_recording_keeps_measured_width_blank_and_saves_reference(self):
        hand, mask, segments = fixture()
        frame = np.full((384, 320, 3), 165, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(close=lambda: None)
            session = RealtimeSession(model, None, Path(directory), {'reference_grip_kgf': 31.5}, 'test')
            band = AxisBottleTracker(frame, POSE.cap, POSE.base, [], 120.)
            session.band = band

            def extract(*_):
                band.outline_segments = segments
                empty = np.full(31, np.nan)
                return empty.copy(), empty.copy(), np.ones(31), np.ones(31)

            try:
                with patch('main.observe_hand', return_value=Hand('Right', hand.astype(int).tolist(), np.full(14, 150.))), \
                        patch.object(band.endpoint_tracker, 'track', return_value=(POSE, 1., 'Axis OK')), \
                        patch('bottle_axis.hand_mask', return_value=mask), \
                        patch('bottle_axis.body_occlusion_mask', return_value=mask), \
                        patch.object(band, '_extract', side_effect=extract):
                    session.process(FramePacket(0, 100., frame), ['h'])
                    for tick, stamp in enumerate((3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4), 1):
                        result = session.process(FramePacket(tick, 100 + stamp, frame), [])
                    self.assertTrue(result['joint_estimate_ready'])
                    self.assertFalse(result['baseline_ready'])
                    session.process(FramePacket(9, 104.6, frame), ['r'])
                    self.assertIsNotNone(session.writer)
                    session.process(FramePacket(10, 104.7, frame), ['h'])
                    self.assertTrue(band.joint_estimator.ready)
                    self.assertFalse(band.joint_estimator.calibrating)
                    with patch.object(band.endpoint_tracker, 'track', return_value=(None, 0., 'Axis lost')):
                        lost = session.process(FramePacket(11, 104.8, frame), [])
                    self.assertIsNone(lost['joint_estimated_width_px'])
                    snapshot = save_debug_snapshot(Path(directory), FramePacket(11, 104.8, frame), lost, 104.8)
                    session.process(FramePacket(12, 104.9, frame), ['r'])
                csv_path = next(Path(directory).glob('*.csv'))
                with csv_path.open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(None not in row for row in rows))
                self.assertTrue(all(row['outline_visible'] == '0' and row['bottle_width_px'] == '' for row in rows))
                self.assertTrue(all(row['median_width_reduction_pct'] == '' for row in rows))
                self.assertAlmostEqual(float(rows[0]['joint_estimated_width_px']), 120.)
                self.assertEqual(rows[-1]['joint_estimated_width_px'], '')
                self.assertEqual(len(json.loads(rows[0]['joint_estimated_left_canonical_px'])), 31)
                context = json.loads(csv_path.with_suffix('.json').read_text())
                self.assertEqual(context['schema_version'], 3)
                self.assertFalse(context['joint_estimate']['is_measured_bottle_width'])
                self.assertIsNone(context['baseline_width_px'])
                state = json.loads(snapshot.with_suffix('.json').read_text())
                self.assertTrue(state['analysis']['joint_estimate_ready'])
                self.assertIsNone(state['analysis']['joint_estimated_width_px'])
                self.assertFalse(state['analysis']['joint_estimate_calibrating'])
                self.assertEqual(state['analysis']['joint_estimate_segments'], 0)
                self.assertIn('axis lost', state['analysis']['joint_estimate_reason'])
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
