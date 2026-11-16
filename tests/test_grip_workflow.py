"""Guided acquisition -> features -> personal reference -> label-free reuse.

Hand observations are controlled test inputs, not an accuracy experiment.
"""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bottle_axis import AxisBottleTracker
from bottle_hand_tracking import GripTrack
from bottle_outline import OutlineMeasurement
from grip_features import frame_features
from grip_runtime import load_settings, label_metadata
from main import RealtimeSession, Hand, FrameData, Baseline
from realtime import FramePacket
from test_joint_estimate import POSE, fixture, calibrated


class GuidedGripTests(unittest.TestCase):
    def test_h_calibration_saves_world_angles_and_features_exclude_flow_only(self):
        hand, mask, segments = fixture()
        estimator = calibrated()
        estimator.request(0.)
        for stamp in (3., 3.2, 3.4, 3.6):
            estimator.update(POSE, hand, segments, mask, stamp, angles=np.full(14, 150.))
        np.testing.assert_allclose(estimator.baseline['angles'], 150.)
        band = SimpleNamespace(joint_estimator=estimator, joint_hand=SimpleNamespace(last=GripTrack(hand.tolist(), 'Right', 'landmarks')),
                               pose=POSE)
        outline = OutlineMeasurement(axis_points=POSE.points, axis_quality=1., metrics={'joint_estimated_width_reduction_pct': 10.})
        data = FrameData(Hand('Right', hand.astype(int).tolist(), np.full(14, 130.)), (None, None), outline)
        values = frame_features(data, band, Baseline())
        self.assertEqual(values['hand_flexion_change_positive_mean_deg'], 20.)
        estimator.display_only = True
        values = frame_features(data, band, Baseline())
        self.assertEqual(values['valid_hand'], 0.)
        self.assertIsNone(values['proxy_width_reduction_pct'])

    def test_first_guided_trial_calibrates_and_second_predicts_without_a_current_label(self):
        hand, _, _ = fixture()
        frame = np.full((384, 320, 3), 165, np.uint8)
        settings = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def run_trial(amplitude, predict_only):
                metadata = label_metadata(settings, predict_only=predict_only)
                model = SimpleNamespace(close=lambda: None)
                session = RealtimeSession(model, None, root / 'recordings', metadata, 'test',
                                          grip_settings=settings, grip_root=root)
                band = AxisBottleTracker(frame, POSE.cap, POSE.base, hand.tolist(), 120.)
                band.joint_estimator = calibrated()
                band.joint_estimator.baseline['angles'] = np.full(14, 150.)
                band.joint_hand.last = GripTrack(hand.tolist(), 'Right', 'landmarks')
                session.band = band

                def observe(*args):
                    now = args[2] / 1000
                    phase = session.grip.phase(now)
                    change = amplitude if phase == 'squeeze' else 0.
                    outline = OutlineMeasurement(axis_points=POSE.points, axis_quality=1., band_widths=(None,) * 31,
                                                 metrics={'joint_estimated_width_reduction_pct': change / 2})
                    band.last = outline
                    return FrameData(Hand('Right', hand.astype(int).tolist(), np.full(14, 150. - change)),
                                     (None, None), outline)

                try:
                    with patch('main.observe', side_effect=observe):
                        for i in range(85):
                            session.process(FramePacket(i, 100 + i / 12, frame), ['g'] if i == 0 else [])
                        self.assertIsNone(session.writer)
                        self.assertTrue(session.grip.last_record['quality']['usable_for_model'], session.grip.last_record['quality'])
                        self.assertTrue(session.grip.reference_consumed)
                        result = session.grip.last_result
                        session.process(FramePacket(90, 107.5, frame), ['g'])
                        self.assertIsNone(session.writer, 'A new trial must request a new relaxed reference')
                    return result
                finally:
                    session.close()

            first = run_trial(20., predict_only=False)
            self.assertEqual(first['status'], 'personal_calibration_saved')
            self.assertEqual(first['reference_kgf'], 80.)
            self.assertIsNone(first['kgf'])
            second = run_trial(10., predict_only=True)
            self.assertEqual(second['status'], 'personal_scaling_demo')
            self.assertAlmostEqual(second['kgf'], 40.)
            files = sorted((root / 'recordings').glob('outline_*.csv'))
            raw = [f for f in files if '.features' not in f.stem]
            with raw[-1].open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(r['reference_grip_kgf'] == '' for r in rows))
            self.assertEqual({r['trial_phase'] for r in rows}, {'rest', 'squeeze', 'release'})
            report = json.loads(raw[-1].with_suffix('.estimate.json').read_text())
            self.assertFalse(report['accuracy_validated'])


if __name__ == '__main__':
    unittest.main()
