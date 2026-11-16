"""Known waveforms verify features; these are not real grip-accuracy data."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grip_features import FRAME_NAMES, angle_features, row_features, summarize_rows, extract_recording
from grip_runtime import load_settings, label_metadata, GripExperiment
from grip_model import calibrate_personal, predict_personal


def trial_rows(amplitude=20., participant='p1', target=80., source='dynamometer'):
    rows = []
    for i in range(85):
        t = i / 12
        phase = 'rest' if t < 2 else 'squeeze' if t < 5 else 'release'
        strength = amplitude if phase == 'squeeze' else 0.
        row = {f'feat_{n}': None for n in FRAME_NAMES}
        row.update(time_s=t, source_frame_id=i, trial_phase=phase, trial_kind='maximal',
                   participant_id=participant, bottle_id='same_pet_empty_capped', tested_hand='right',
                   reference_grip_kgf=target, label_source=source,
                   feat_hand_flexion_change_positive_mean_deg=strength,
                   feat_hand_flexion_change_mean_deg=strength,
                   feat_observed_width_reduction_pct=strength / 4,
                   feat_proxy_width_reduction_pct=strength / 2,
                   feat_valid_hand=1, feat_valid_observed=1, feat_valid_proxy=1, feat_valid_axis=1,
                   joint_display_only=False, joint_hand_source='landmarks')
        rows.append(row)
    return rows


def synthetic_trial(amplitude=20., participant='p1', target=80., source='dynamometer'):
    return summarize_rows(trial_rows(amplitude, participant, target, source), participant + '_trial')


class GripFeatureTests(unittest.TestCase):
    def test_flexion_sign_and_trial_waveform_have_known_features(self):
        f = angle_features(np.full(14, 130.), np.full(14, 150.))
        self.assertEqual(f['hand_flexion_change_mean_deg'], 20.)
        record = synthetic_trial()
        self.assertTrue(record['quality']['usable_for_model'], record['quality'])
        self.assertEqual(record['features']['hand_flexion_change_positive_mean_deg_p90'], 20.)
        self.assertEqual(record['features']['observed_width_reduction_pct_p90'], 5.)
        self.assertEqual(record['features']['proxy_width_reduction_pct_p90'], 10.)
        self.assertEqual(record['features']['proxy_width_reduction_pct_release_residual'], 0.)
        self.assertFalse(record['label_is_frame_force'])

    def test_static_video_and_flow_only_frames_cannot_become_training_trials(self):
        static = synthetic_trial(amplitude=0.)
        self.assertFalse(static['quality']['usable_for_model'])
        rows = trial_rows()
        for row in rows:
            row['joint_display_only'] = True
        record = summarize_rows(rows)
        self.assertFalse(record['quality']['usable_for_model'])
        self.assertIsNone(record['features']['hand_flexion_change_positive_mean_deg_p90'])
        self.assertEqual(row_features(rows[0])['valid_hand'], 0.)

    def test_a_hand_gesture_without_a_supported_bottle_axis_is_not_a_grip_trial(self):
        rows = trial_rows()
        for row in rows:
            row['feat_valid_axis'] = 0
        record = summarize_rows(rows)
        self.assertFalse(record['quality']['usable_for_model'])
        self.assertIn('bottle_axis_missing_too_often', record['quality']['issues'])

    def test_duplicate_frames_do_not_increase_sample_count(self):
        rows = trial_rows()
        result = summarize_rows(rows + rows)
        self.assertEqual(result['quality']['total_frames'], 85)

    def test_plastic_deformation_does_not_require_an_elastic_bottle_return(self):
        rows = trial_rows()
        for row in rows:
            if row['trial_phase'] == 'release':
                row['feat_hand_flexion_change_positive_mean_deg'] = 20.
                row['feat_observed_width_reduction_pct'] = 5.
                row['feat_proxy_width_reduction_pct'] = 10.
        result = summarize_rows(rows)
        self.assertTrue(result['quality']['usable_for_model'])
        self.assertIn('no_visible_return_after_release', result['quality']['warnings'])
        self.assertEqual(result['features']['observed_width_reduction_pct_release_residual'], 5.)

    def test_label_conflicts_and_interrupted_sidecars_are_rejected(self):
        rows = trial_rows()
        rows[10]['reference_grip_kgf'] = 60.
        with self.assertRaises(ValueError):
            summarize_rows(rows)
        rows = trial_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trial.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            path.with_suffix('.json').write_text(json.dumps({'grip_trial_guided': True, 'grip_trial_completed': False}))
            result = extract_recording(path)
            self.assertFalse(result['quality']['usable_for_model'])
            self.assertIn('guided_trial_interrupted', result['quality']['issues'])
            self.assertTrue(path.with_suffix('.features.csv').is_file())

    def test_provisional_80_applies_only_to_the_default_person(self):
        settings = load_settings()
        own = label_metadata(settings)
        self.assertEqual(own['reference_grip_kgf'], 80.)
        self.assertEqual(own['label_source'], 'provisional_user_assumption')
        other = label_metadata(settings, participant='someone_else')
        self.assertEqual(other['reference_grip_kgf'], '')
        measured = label_metadata(settings, participant='p2', grip=42.)
        self.assertEqual(measured['reference_grip_kgf'], 42.)
        self.assertEqual(measured['label_source'], 'dynamometer')
        self.assertEqual(label_metadata(settings, predict_only=True)['reference_grip_kgf'], '')

    def test_personal_calibration_is_not_a_fitted_accuracy_claim_or_label_leak(self):
        reference = synthetic_trial(source='provisional_user_assumption')
        artifact = calibrate_personal(reference)
        current = synthetic_trial(amplitude=10., target=5., source='unspecified')
        current['recording_id'] = 'new_unlabeled_trial'
        prediction = predict_personal(artifact, current)
        self.assertEqual(prediction['kgf'], 40.)
        self.assertEqual(prediction['operation_comparison_kgf']['square_ratio'], 20.)
        current['reference_grip_kgf'] = 999.
        self.assertEqual(predict_personal(artifact, current)['kgf'], 40.)
        self.assertFalse(prediction['accuracy_validated'])
        self.assertIsNone(predict_personal(artifact, reference)['kgf'])
        current['participant_id'] = 'p2'
        self.assertIsNone(predict_personal(artifact, current)['kgf'])

    def test_guided_timing_is_independent_of_frame_rate(self):
        run = GripExperiment(label_metadata(load_settings()))
        run.begin(10., guided=True)
        self.assertEqual(run.phase(11.99), 'rest')
        self.assertEqual(run.phase(12.), 'squeeze')
        self.assertEqual(run.phase(15.), 'release')
        self.assertTrue(run.due(17.))

    def test_actual_dynamometer_label_replaces_a_provisional_personal_reference(self):
        settings = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (target, source) in enumerate(((80., 'provisional_user_assumption'), (65., 'dynamometer'))):
                rows = trial_rows(target=target, source=source)
                path = root / f'trial{index}.csv'
                with path.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                run = GripExperiment({}, settings, root=root)
                run.begin(0., guided=True)
                result = run.finish(path, completed=True)
                self.assertIsNone(result['kgf'])
                self.assertEqual(result['reference_kgf'], target)
                self.assertEqual(result['status'], 'personal_calibration_saved')


if __name__ == '__main__':
    unittest.main()
