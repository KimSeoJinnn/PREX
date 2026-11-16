"""Small-sample, participant-separated regression using only local NumPy.

Feature selection, imputation, scaling and model choice are fitted INSIDE each
training fold. Labels/IDs/quality flags are never predictor columns. Temporary
80 kgf assumptions are excluded from population training by default.
"""

import json
from pathlib import Path

import numpy as np

from grip_features import SIGNALS, EVENT_SIGNALS, number


ALLOWED_FEATURES = ({s + suffix for s in SIGNALS for suffix in ('_p50', '_p90', '_p10', '_range')}
                    | {signal + suffix for signal, _ in EVENT_SIGNALS.values()
                       for suffix in ('_release_residual', '_rise_speed_p90', '_time_to_peak_s',
                                      '_hold_fraction_above_half')})
CANDIDATES = [
    {'kind': 'mean', 'group': 'none'},
    *[{'kind': 'ridge', 'group': 'fusion', 'alpha': a} for a in (1., 10., 100.)],
    {'kind': 'nonlinear_ridge', 'group': 'fusion', 'alpha': 10.},
    *[{'kind': 'ridge', 'group': g, 'alpha': 10.} for g in ('hand', 'observed', 'proxy')],
    {'kind': 'rbf_ridge', 'group': 'fusion', 'alpha': 1.},
    {'kind': 'knn', 'group': 'fusion', 'k': 3},
]


def ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - np.mean(x), y - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-10 else None


def prepare_subjects(records, include_provisional=False, bottle_id=None):
    groups, excluded, seen = {}, [], set()
    for record in records:
        reason = None
        target = number(record.get('reference_grip_kgf'))
        digest = record.get('source_sha256')
        if digest and digest in seen:
            reason = 'duplicate_recording'
        elif not record.get('quality', {}).get('usable_for_model'):
            reason = 'recording_quality_or_protocol'
        elif not record.get('participant_id'):
            reason = 'missing_participant_id'
        elif not record.get('bottle_id'):
            reason = 'missing_bottle_condition'
        elif target is None or target <= 0:
            reason = 'missing_reference_label'
        elif record.get('label_source') != 'dynamometer' and not include_provisional:
            reason = 'label_not_dynamometer_measured'
        elif bottle_id and record.get('bottle_id') != bottle_id:
            reason = 'different_bottle_condition'
        if reason:
            excluded.append({'recording_id': record.get('recording_id'), 'reason': reason})
            continue
        if digest:
            seen.add(digest)
        key = (record['participant_id'], record['bottle_id'], record.get('tested_hand', 'unspecified'))
        groups.setdefault(key, []).append(record)
    subjects = []
    for (participant, bottle, hand), trials in groups.items():
        names = sorted(set().union(*(r['features'].keys() for r in trials)) & ALLOWED_FEATURES)
        features = {}
        for name in names:
            values = [number(r['features'].get(name)) for r in trials]
            values = [v for v in values if v is not None]
            features[name] = float(np.median(values)) if values else None
        subjects.append({'participant_id': participant, 'bottle_id': bottle, 'tested_hand': hand,
                         'reference_grip_kgf': float(np.median([r['reference_grip_kgf'] for r in trials])),
                         'features': features, 'trials': len(trials),
                         'label_sources': sorted({r['label_source'] for r in trials})})
    return subjects, excluded


def correlations(subjects, permutations=499, seed=42):
    names = sorted(set().union(*(r['features'] for r in subjects))) if subjects else sorted(ALLOWED_FEATURES)
    rng = np.random.default_rng(seed)
    result = []
    for name in names:
        pairs = [(number(r['features'].get(name)), number(r['reference_grip_kgf'])) for r in subjects]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        row = {'feature': name, 'n_participants': len(pairs), 'pearson_r': None, 'spearman_r': None,
               'permutation_p': None, 'fdr_q': None, 'reason': ''}
        if len(pairs) < 5:
            row['reason'] = 'need_at_least_5_independent_participants'
        else:
            x, y = np.array(pairs).T
            if np.std(y) < 1e-8:
                row['reason'] = 'constant_grip_labels_correlation_undefined'
            elif np.std(x) < 1e-8:
                row['reason'] = 'constant_feature'
            else:
                row['pearson_r'] = correlation(x, y)
                rx, ry = ranks(x), ranks(y)
                observed = correlation(rx, ry)
                row['spearman_r'] = observed
                null = [abs(correlation(rx, rng.permutation(ry)) or 0.) for _ in range(permutations)]
                row['permutation_p'] = (1 + sum(v >= abs(observed) - 1e-12 for v in null)) / (permutations + 1)
        result.append(row)
    tested = sorted((r for r in result if r['permutation_p'] is not None), key=lambda r: r['permutation_p'])
    last = 1.
    for i in range(len(tested) - 1, -1, -1):
        last = min(last, tested[i]['permutation_p'] * len(tested) / (i + 1))
        tested[i]['fdr_q'] = float(last)
    return result


def _group_feature(name, group):
    if group == 'fusion':
        return True
    if group == 'hand':
        return name.startswith(('hand_', 'angle_', 'flexion_change_'))
    return name.startswith(group + '_')


def _matrix(records, names):
    return np.array([[np.nan if number(r['features'].get(n)) is None else float(r['features'][n])
                      for n in names] for r in records], float)


def _expand(z, kind):
    if kind != 'nonlinear_ridge':
        return z
    columns = [z, z**2, np.sign(z) * np.sqrt(np.abs(z)), np.sign(z) * np.log1p(np.abs(z))]
    interactions = [z[:, i] * z[:, j] for i in range(min(z.shape[1], 4)) for j in range(i + 1, min(z.shape[1], 4))]
    if interactions:
        columns.append(np.column_stack(interactions))
    return np.column_stack(columns)


def fit(records, candidate):
    y = np.array([r['reference_grip_kgf'] for r in records], float)
    model = {'kind': 'mean', 'mean_y': float(np.mean(y)), 'names': [], 'candidate': candidate}
    if candidate['kind'] == 'mean':
        return model
    names = sorted(set().union(*(r['features'] for r in records)) & ALLOWED_FEATURES)
    names = [n for n in names if _group_feature(n, candidate['group'])]
    if not names:
        return model
    x = _matrix(records, names)
    usable = np.mean(np.isfinite(x), axis=0) >= .7
    names = [n for n, ok in zip(names, usable) if ok]
    if not names:
        return model
    x = x[:, usable]
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, median)
    std = np.std(x, axis=0)
    good = std > 1e-8
    if not np.any(good):
        return model
    x, median, std = x[:, good], median[good], std[good]
    names = [n for n, ok in zip(names, good) if ok]
    scores = [abs(correlation(ranks(x[:, i]), ranks(y)) or 0.) for i in range(len(names))]
    count = min(6, max(1, len(records) // 3), len(names))
    selected = np.argsort(-np.array(scores), kind='stable')[:count]
    names = [names[i] for i in selected]
    median, std = median[selected], std[selected]
    x = x[:, selected]
    center = np.mean(x, axis=0)
    z = np.clip((x - center) / std, -5, 5)
    model.update(kind=candidate['kind'], names=names, median=median.tolist(), center=center.tolist(), scale=std.tolist())
    if candidate['kind'] in ('ridge', 'nonlinear_ridge'):
        expanded = _expand(z, candidate['kind'])
        expanded_center = np.mean(expanded, axis=0)
        expanded -= expanded_center
        weights = np.linalg.solve(expanded.T @ expanded + candidate['alpha'] * np.eye(expanded.shape[1]),
                                  expanded.T @ (y - np.mean(y)))
        model.update(weights=weights.tolist(), expanded_center=expanded_center.tolist())
    elif candidate['kind'] == 'rbf_ridge':
        distances = np.sum((z[:, None, :] - z[None, :, :])**2, axis=2) / len(names)
        weights = np.linalg.solve(np.exp(-distances) + candidate['alpha'] * np.eye(len(y)), y - np.mean(y))
        model.update(train_z=z.tolist(), weights=weights.tolist())
    else:
        model.update(train_z=z.tolist(), train_y=y.tolist(), k=candidate['k'])
    return model


def predict_values(model, records):
    if model['kind'] == 'mean':
        return np.full(len(records), model['mean_y'])
    x = _matrix(records, model['names'])
    x = np.where(np.isfinite(x), x, np.array(model['median']))
    z = np.clip((x - np.array(model['center'])) / np.array(model['scale']), -5, 5)
    kind = model['kind']
    if kind in ('ridge', 'nonlinear_ridge'):
        expanded = _expand(z, kind) - np.array(model['expanded_center'])
        return model['mean_y'] + expanded @ np.array(model['weights'])
    train = np.array(model['train_z'])
    distances = np.sum((z[:, None, :] - train[None, :, :])**2, axis=2) / len(model['names'])
    if kind == 'rbf_ridge':
        return model['mean_y'] + np.exp(-distances) @ np.array(model['weights'])
    result = []
    for d in distances:
        indices = np.argsort(d)[:min(model['k'], len(train))]
        weights = 1 / np.maximum(d[indices], 1e-6)
        result.append(float(np.average(np.array(model['train_y'])[indices], weights=weights)))
    return np.array(result)


def _folds(records, folds=5, seed=42):
    groups = sorted({r['participant_id'] for r in records})
    np.random.default_rng(seed).shuffle(groups)
    for held in np.array_split(groups, min(folds, len(groups))):
        held = set(held)
        train = [i for i, r in enumerate(records) if r['participant_id'] not in held]
        test = [i for i, r in enumerate(records) if r['participant_id'] in held]
        if train and test:
            yield train, test


def choose(records):
    scores = []
    for candidate in CANDIDATES:
        errors = []
        for train, test in _folds(records, folds=3, seed=17):
            model = fit([records[i] for i in train], candidate)
            predictions = predict_values(model, [records[i] for i in test])
            errors.extend(abs(predictions - [records[i]['reference_grip_kgf'] for i in test]))
        scores.append(float(np.mean(errors)) if errors else float('inf'))
    return CANDIDATES[int(np.argmin(scores))], scores


def _metrics(y, predictions):
    error = predictions - y
    denominator = float(np.sum((y - np.mean(y))**2))
    return {'mae_kgf': float(np.mean(np.abs(error))), 'rmse_kgf': float(np.sqrt(np.mean(error**2))),
            'r2': float(1 - np.sum(error**2) / denominator) if denominator > 1e-8 else None}


def train_evaluate(subjects):
    report = {'status': 'insufficient_data', 'n_participants': len({r['participant_id'] for r in subjects}),
              'clinical_validated': False, 'method': 'nested_participant_group_cv',
              'target': 'maximal_dynamometer_grip_kgf_per_trial_not_frame_force',
              'method_source': 'https://arxiv.org/abs/1606.05201'}
    if len({(r['bottle_id'], r.get('tested_hand')) for r in subjects}) > 1:
        report.update(status='mixed_conditions', reason='Select one bottle condition and tested hand for one model')
        return report, None
    if report['n_participants'] < 6:
        report['reason'] = 'Need at least 6 independent participants for exploratory model comparison; this is not a clinical sample-size guarantee'
        return report, None
    y = np.array([r['reference_grip_kgf'] for r in subjects], float)
    if len(np.unique(y)) < 3:
        report.update(status='constant_or_insufficient_label_variation', reason='A single 80 kgf label cannot identify a force mapping')
        return report, None
    oof = np.full(len(y), np.nan)
    comparisons = [np.full(len(y), np.nan) for _ in CANDIDATES]
    splits = []
    for train, test in _folds(subjects, folds=min(5, max(3, len(subjects) // 2)), seed=42):
        training, testing = [subjects[i] for i in train], [subjects[i] for i in test]
        candidate, _ = choose(training)
        oof[test] = predict_values(fit(training, candidate), testing)
        for j, fixed in enumerate(CANDIDATES):
            comparisons[j][test] = predict_values(fit(training, fixed), testing)
        splits.append({'training_participants': [r['participant_id'] for r in training],
                       'test_participants': [r['participant_id'] for r in testing], 'selected': candidate})
    baseline = _metrics(y, comparisons[0])
    evaluation = _metrics(y, oof)
    rng = np.random.default_rng(42)
    absolute_error = np.abs(oof - y)
    boot = [float(np.mean(absolute_error[rng.integers(0, len(y), len(y))])) for _ in range(500)]
    selected, inner_scores = choose(subjects)
    final = fit(subjects, selected)
    improves = evaluation['mae_kgf'] < baseline['mae_kgf'] * .95 and final['kind'] != 'mean'
    report.update(status='research_model_available' if improves else 'no_reliable_improvement_over_mean',
                  nested_oof_metrics=evaluation, mean_baseline_metrics=baseline,
                  mae_bootstrap_subject_interval_95=np.percentile(boot, [2.5, 97.5]).tolist(),
                  interval_scope='descriptive_OOF_error_not_individual_prediction_interval',
                  fixed_candidate_comparisons=[{'candidate': c, **_metrics(y, p)} for c, p in zip(CANDIDATES, comparisons)],
                  full_data_inner_selection_mae=[{'candidate': c, 'mae_kgf': s} for c, s in zip(CANDIDATES, inner_scores)],
                  folds=splits, oof_predictions=[{'participant_id': r['participant_id'], 'truth_kgf': float(a),
                                                'prediction_kgf': float(b)} for r, a, b in zip(subjects, y, oof)],
                  selected_candidate=selected, selected_features=final['names'])
    if not improves:
        return report, None
    artifact = {'schema_version': 1, 'kind': 'population_grip_research', 'clinical_validated': False,
                'feature_version': 'grip_trial_v1', 'bottle_id': subjects[0]['bottle_id'],
                'tested_hand': subjects[0].get('tested_hand', 'unspecified'),
                'label_sources': sorted(set().union(*(r['label_sources'] for r in subjects))),
                'training_participants': sorted({r['participant_id'] for r in subjects}),
                'training_label_range_kgf': [float(np.min(y)), float(np.max(y))],
                'evaluation': report, 'model': final}
    return report, artifact


def predict_trial(artifact, record):
    if artifact.get('kind') != 'population_grip_research' or artifact.get('feature_version') != record.get('feature_version'):
        return {'status': 'incompatible_model', 'kgf': None}
    if not record.get('quality', {}).get('usable_for_model'):
        return {'status': 'insufficient_trial_quality', 'kgf': None}
    if any(artifact.get(k) != record.get(k) for k in ('bottle_id', 'tested_hand')):
        return {'status': 'different_bottle_or_hand_condition', 'kgf': None}
    model = artifact['model']
    if model['kind'] == 'mean' or not model['names'] or any(n not in ALLOWED_FEATURES for n in model['names']):
        return {'status': 'no_feature_based_model', 'kgf': None}
    x = _matrix([record], model['names'])[0]
    valid = np.isfinite(x)
    if np.mean(valid) < .6:
        return {'status': 'missing_required_features', 'kgf': None}
    z = np.abs((x[valid] - np.array(model['center'])[valid]) / np.array(model['scale'])[valid])
    if np.max(z) > 6:
        return {'status': 'outside_feature_training_range', 'kgf': None}
    # No participant ID or current ground-truth label enters this calculation.
    value = float(predict_values(model, [record])[0])
    return {'status': 'research_estimate' if value > 0 else 'invalid_estimate',
            'kgf': value if value > 0 else None, 'clinical_validated': False,
            'participant_seen_in_training': record.get('participant_id') in artifact.get('training_participants', []),
            'cross_validated_mae_kgf': artifact['evaluation']['nested_oof_metrics']['mae_kgf']}


PERSONAL_CHANNELS = {
    'hand': ('hand_flexion_change_positive_mean_deg_p90', 3.),
    'observed': ('observed_width_reduction_pct_p90', 1.),
    'proxy': ('proxy_width_reduction_pct_p90', 1.5),
}


def calibrate_personal(record):
    target = number(record.get('reference_grip_kgf'))
    if not record.get('quality', {}).get('usable_for_model') or target is None or target <= 0:
        raise ValueError('A supported maximal squeeze-release trial and a positive reference label are required')
    channels = {group: {'feature': name, 'reference': float(record['features'][name])}
                for group, (name, minimum) in PERSONAL_CHANNELS.items()
                if (number(record['features'].get(name)) or 0) >= minimum}
    if not channels:
        raise ValueError('No sufficiently strong supported feature channel for personal calibration')
    return {'schema_version': 1, 'kind': 'personal_reference_scaling_demo',
            'participant_id': record['participant_id'], 'bottle_id': record['bottle_id'],
            'tested_hand': record.get('tested_hand', 'unspecified'), 'feature_version': record['feature_version'],
            'reference_grip_kgf': target, 'label_source': record.get('label_source', 'unspecified'),
            'reference_recording_id': record['recording_id'], 'channels': channels,
            'reference_source_sha256': record.get('source_sha256'),
            'accuracy_validated': False, 'is_instantaneous_force': False,
            'assumption': 'one_point_scaling_only; nonlinear_choices_not_identifiable_from_one_label'}


def predict_personal(artifact, record):
    if any(artifact.get(k) != record.get(k) for k in ('participant_id', 'bottle_id', 'tested_hand', 'feature_version')):
        return {'status': 'different_person_or_condition', 'kgf': None}
    if not record.get('quality', {}).get('usable_for_model'):
        return {'status': 'insufficient_trial_quality', 'kgf': None}
    if (record.get('recording_id') == artifact.get('reference_recording_id')
            or (artifact.get('reference_source_sha256')
                and artifact['reference_source_sha256'] == record.get('source_sha256'))):
        return {'status': 'calibration_reference_replay_not_prediction', 'kgf': None,
                'reference_kgf': artifact['reference_grip_kgf'], 'accuracy_validated': False}
    ratios = {}
    for family, channel in artifact['channels'].items():
        value = number(record['features'].get(channel['feature']))
        if value is not None and value >= 0:
            ratios[family] = value / channel['reference']
    if not ratios:
        return {'status': 'missing_calibrated_channels', 'kgf': None}
    # The pink proxy comes from the hand, so it is not a second independent
    # vote when an angle-based hand channel is already available.
    independent = [ratios['hand']] if 'hand' in ratios else ([ratios['proxy']] if 'proxy' in ratios else [])
    if 'observed' in ratios:
        independent.append(ratios['observed'])
    ratio = float(np.median(independent))
    if ratio > 2. or ratio < .15:
        return {'status': 'outside_personal_reference_range', 'kgf': None}
    label = artifact['reference_grip_kgf']
    return {'status': 'personal_scaling_demo', 'kgf': label * ratio,
            'relative_response': ratio, 'channel_ratios': ratios,
            'operation_comparison_kgf': {'linear_ratio': label * ratio, 'square_ratio': label * ratio**2,
                                         'sqrt_ratio': label * np.sqrt(ratio), 'log_ratio': label * np.log1p(ratio) / np.log(2)},
            'chosen_operation': 'linear_ratio_by_design_not_by_closeness_to_80',
            'accuracy_validated': False, 'label_source': artifact.get('label_source')}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
