"""Synthetic statistical checks, never a report of camera/HGS accuracy."""

import copy
import unittest

import numpy as np

from grip_model import (ALLOWED_FEATURES, prepare_subjects, correlations, fit, predict_values,
                        train_evaluate, predict_trial)
from test_grip_features import synthetic_trial


class GripModelTests(unittest.TestCase):
    def records(self, count=18):
        return [synthetic_trial(amplitude=(24 + i * 2) / 4, participant=f'p{i}', target=24 + i * 2)
                for i in range(count)]

    def test_many_trials_of_one_person_are_not_many_independent_people(self):
        trials = [synthetic_trial(participant='same') for _ in range(20)]
        subjects, _ = prepare_subjects(trials)
        self.assertEqual(len(subjects), 1)
        report, artifact = train_evaluate(subjects)
        self.assertEqual(report['n_participants'], 1)
        self.assertIsNone(artifact)

    def test_provisional_labels_are_excluded_by_default_and_constant_y_is_undefined(self):
        trials = [synthetic_trial(participant=f'p{i}', amplitude=i + 4, source='provisional_user_assumption')
                  for i in range(8)]
        subjects, excluded = prepare_subjects(trials)
        self.assertEqual(subjects, [])
        self.assertEqual(len(excluded), 8)
        subjects, _ = prepare_subjects(trials, include_provisional=True)
        rows = correlations(subjects, permutations=29)
        self.assertTrue(all(r['pearson_r'] is None and r['spearman_r'] is None for r in rows))
        self.assertTrue(any(r['reason'] == 'constant_grip_labels_correlation_undefined' for r in rows))

    def test_known_monotone_data_produces_correlation_with_independent_subject_count(self):
        subjects, _ = prepare_subjects(self.records(8))
        rows = correlations(subjects, permutations=99)
        row = next(r for r in rows if r['feature'] == 'observed_width_reduction_pct_p90')
        self.assertAlmostEqual(row['spearman_r'], 1.)
        self.assertEqual(row['n_participants'], 8)
        self.assertGreaterEqual(row['fdr_q'], row['permutation_p'])

    def test_no_signal_does_not_publish_a_constant_grip_model(self):
        records = self.records(8)
        for record in records:
            record['features'] = {'hand_flexion_change_positive_mean_deg_p90': 10.}
        subjects, _ = prepare_subjects(records)
        report, artifact = train_evaluate(subjects)
        self.assertIsNone(artifact)
        self.assertEqual(report['status'], 'no_reliable_improvement_over_mean')

    def test_imputation_scaling_selection_and_prediction_do_not_read_test_labels(self):
        subjects, _ = prepare_subjects(self.records(9))
        subjects[0]['features']['reference_grip_kgf'] = 1e9
        model = fit(subjects[:7], {'kind': 'ridge', 'group': 'fusion', 'alpha': 1.})
        self.assertTrue(set(model['names']) <= ALLOWED_FEATURES)
        before = predict_values(model, subjects[7:])
        changed = copy.deepcopy(subjects[7:])
        for r in changed:
            r['reference_grip_kgf'] = -1e9
            r['participant_id'] = 'unseen'
        np.testing.assert_allclose(predict_values(model, changed), before)
        self.assertEqual(model['mean_y'], np.mean([r['reference_grip_kgf'] for r in subjects[:7]]))

    def test_nested_subject_validation_beats_mean_on_a_known_synthetic_relationship(self):
        subjects, _ = prepare_subjects(self.records())
        report, artifact = train_evaluate(subjects)
        self.assertIsNotNone(artifact, report['status'])
        self.assertLess(report['nested_oof_metrics']['mae_kgf'], report['mean_baseline_metrics']['mae_kgf'])
        for fold in report['folds']:
            self.assertFalse(set(fold['training_participants']) & set(fold['test_participants']))
        target = copy.deepcopy(subjects[5])
        target.update(feature_version='grip_trial_v1', quality={'usable_for_model': True})
        first = predict_trial(artifact, target)
        target['reference_grip_kgf'] = None
        self.assertEqual(predict_trial(artifact, target)['kgf'], first['kgf'])
        target['bottle_id'] = 'different'
        self.assertIsNone(predict_trial(artifact, target)['kgf'])
        self.assertFalse(artifact['clinical_validated'])


if __name__ == '__main__':
    unittest.main()
