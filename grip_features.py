"""Video-derived trial features. A dynamometer maximum is a TRIAL label.

Observed bottle deformation and the hand-derived pink proxy stay separate.
Optical-flow-only frames supply no numeric features. Nothing here converts
camera measurements to force without an explicitly fitted/calibrated model.
"""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


JOINT_NAMES = ('thumb_mcp', 'thumb_ip', 'index_mcp', 'index_pip', 'index_dip',
               'middle_mcp', 'middle_pip', 'middle_dip', 'ring_mcp', 'ring_pip',
               'ring_dip', 'pinky_mcp', 'pinky_pip', 'pinky_dip')
FINGERS = {'thumb': (0, 2), 'index': (2, 5), 'middle': (5, 8), 'ring': (8, 11), 'pinky': (11, 14)}
OBSERVED = {
    'observed_width_reduction_pct': 'median_width_reduction_pct',
    'observed_dent_p95_pct': 'dent_p95_pct',
    'observed_area_reduction_pct': 'visible_area_reduction_pct',
    'observed_left_dent_pct': 'left_dent_p95_pct',
    'observed_right_dent_pct': 'right_dent_p95_pct',
    'observed_contour_rms_pct': 'contour_rms_pct',
}
GEOMETRY = ('hand_hull_area_norm', 'hand_tip_distance_norm', 'hand_thumb_index_norm',
            'hand_thumb_middle_norm', 'hand_tip_span_norm', 'hand_tip_distance_change_norm')
FRAME_NAMES = (
    *[f'angle_{n}_deg' for n in JOINT_NAMES],
    *[f'flexion_change_{n}_deg' for n in JOINT_NAMES],
    'hand_flexion_mean_deg', 'hand_flexion_change_mean_deg',
    'hand_flexion_change_abs_mean_deg', 'hand_flexion_change_positive_mean_deg',
    *[f'hand_{n}_change_deg' for n in FINGERS], *GEOMETRY, *OBSERVED,
    'observed_asymmetry_pct', 'observed_profile_iqr_pct',
    'proxy_width_reduction_pct', 'proxy_curvature_norm',
    'valid_hand', 'valid_observed', 'valid_proxy', 'valid_axis',
)
SIGNALS = tuple(n for n in FRAME_NAMES if not n.startswith('valid_'))
EVENT_SIGNALS = {'hand': ('hand_flexion_change_positive_mean_deg', 3.),
                 'observed': ('observed_width_reduction_pct', 1.),
                 'proxy': ('proxy_width_reduction_pct', 1.5)}


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    return str(value).lower() in ('true', '1', 'yes')


def vector(value, length=None):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        result = np.asarray([np.nan if v is None else float(v) for v in value], float)
        if length is not None and result.shape != (length,):
            return None
        return result
    except (ValueError, TypeError):
        return None


def geometry(points):
    points = np.asarray(points, np.float32)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return {}
    palm = float(np.linalg.norm(points[5] - points[17]))
    if palm < 12:
        return {}
    center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    return {'hand_hull_area_norm': float(cv2.contourArea(cv2.convexHull(points))) / palm**2,
            'hand_tip_distance_norm': float(np.mean(np.linalg.norm(points[[4, 8, 12, 16, 20]] - center, axis=1))) / palm,
            'hand_thumb_index_norm': float(np.linalg.norm(points[4] - points[8])) / palm,
            'hand_thumb_middle_norm': float(np.linalg.norm(points[4] - points[12])) / palm,
            'hand_tip_span_norm': float(np.linalg.norm(points[8] - points[20])) / palm}


def angle_features(angles, reference):
    result = {}
    angles = vector(angles, 14)
    if angles is None or not np.isfinite(angles).all() or np.any((angles < 0) | (angles > 180)):
        return result
    result.update({f'angle_{name}_deg': float(a) for name, a in zip(JOINT_NAMES, angles)})
    result['hand_flexion_mean_deg'] = float(np.mean(180 - angles))
    reference = vector(reference, 14)
    if reference is not None and np.isfinite(reference).all():
        change = reference - angles
        result.update({f'flexion_change_{name}_deg': float(a) for name, a in zip(JOINT_NAMES, change)})
        result.update(hand_flexion_change_mean_deg=float(np.mean(change)),
                      hand_flexion_change_abs_mean_deg=float(np.mean(np.abs(change))),
                      hand_flexion_change_positive_mean_deg=float(np.mean(np.maximum(0, change))))
        for finger, (lo, hi) in FINGERS.items():
            result[f'hand_{finger}_change_deg'] = float(np.mean(change[lo:hi]))
    return result


def frame_features(data, band, baseline):
    result = dict.fromkeys(FRAME_NAMES)
    result.update(valid_hand=0., valid_observed=0., valid_proxy=0., valid_axis=0.)
    outline = data.outline
    metrics = outline.metrics if outline else {}
    axis_ok = outline is not None and outline.axis_points is not None and (number(outline.axis_quality) or 0) >= .5
    result['valid_axis'] = float(axis_ok)
    estimator = getattr(band, 'joint_estimator', None)
    track = getattr(getattr(band, 'joint_hand', None), 'last', None)
    supported = not (estimator and estimator.display_only) and (track is None or track.source == 'landmarks')
    reference = baseline.angles
    joint_reference = estimator.baseline if estimator and estimator.ready else None
    if joint_reference and joint_reference.get('angles') is not None:
        reference = joint_reference['angles']
    if supported and data.hand is not None:
        values = angle_features(data.hand.angles, reference)
        result.update(values)
        result['valid_hand'] = float(bool(values))
        result.update(geometry(data.hand.points))
        if joint_reference and hasattr(band, 'pose'):
            current = cv2.transform(np.asarray(data.hand.points, np.float32).reshape(-1, 1, 2), band.pose.matrix).reshape(-1, 2)
            before, after = geometry(joint_reference['hand']), geometry(current)
            if before and after:
                result['hand_tip_distance_change_norm'] = before['hand_tip_distance_norm'] - after['hand_tip_distance_norm']
    if supported and axis_ok:
        for dest, source in OBSERVED.items():
            result[dest] = number(metrics.get(source))
        result['valid_observed'] = float(result['observed_width_reduction_pct'] is not None)
        left, right = result.get('observed_left_dent_pct'), result.get('observed_right_dent_pct')
        if left is not None and right is not None:
            result['observed_asymmetry_pct'] = abs(left - right)
        if result['valid_observed'] and getattr(band, 'baseline_left', None) is not None:
            l, r = vector(outline.profile_left, 31), vector(outline.profile_right, 31)
            ref_width = band.baseline_right - band.baseline_left
            if l is not None and r is not None:
                valid = np.isfinite(l) & np.isfinite(r) & np.isfinite(ref_width) & (ref_width > 20)
                if np.count_nonzero(valid) >= 6:
                    reductions = 100 * (1 - (r - l)[valid] / ref_width[valid])
                    result['observed_profile_iqr_pct'] = float(np.percentile(reductions, 75) - np.percentile(reductions, 25))
        result['proxy_width_reduction_pct'] = number(metrics.get('joint_estimated_width_reduction_pct'))
        result['valid_proxy'] = float(result['proxy_width_reduction_pct'] is not None)
        if result['valid_proxy'] and estimator:
            widths = estimator.profile_right - estimator.profile_left
            valid = np.isfinite(widths)
            second = np.diff(widths, 2)
            usable = valid[:-2] & valid[1:-1] & valid[2:]
            if np.count_nonzero(usable) >= 3:
                result['proxy_curvature_norm'] = float(np.median(np.abs(second[usable])) / np.nanmedian(widths))
    return {k: number(v) for k, v in result.items()}


def row_features(row):
    if truth(row.get('joint_display_only')):
        result = dict.fromkeys(FRAME_NAMES)
        result.update(valid_hand=0., valid_observed=0., valid_proxy=0.,
                      valid_axis=number(row.get('feat_valid_axis')) or float((number(row.get('axis_quality')) or 0) >= .5))
        return result
    if any('feat_' + n in row for n in FRAME_NAMES):
        result = {n: number(row.get('feat_' + n)) for n in FRAME_NAMES}
        if result['valid_axis'] is None:
            result['valid_axis'] = float((number(row.get('axis_quality')) or 0) >= .5)
        return result
    result = dict.fromkeys(FRAME_NAMES)
    result.update(valid_hand=0., valid_observed=0., valid_proxy=0.,
                  valid_axis=float((number(row.get('axis_quality')) or 0) >= .5))
    # Legacy recordings have angles/deltas but no 2-D hand geometry. Missing
    # fields remain missing; old recordings are never retroactively labeled 80.
    if not truth(row.get('joint_display_only')) and row.get('joint_hand_source', 'landmarks') == 'landmarks':
        angles = [number(row.get(n + '_deg')) for n in JOINT_NAMES]
        delta = [number(row.get(n + '_delta_deg')) for n in JOINT_NAMES]
        if all(v is not None for v in angles):
            ref = np.array(angles) - delta if all(v is not None for v in delta) else None
            result.update(angle_features(angles, ref))
            result['valid_hand'] = 1.
        for dest, source in OBSERVED.items():
            result[dest] = number(row.get(source))
        result['valid_observed'] = float(result['observed_width_reduction_pct'] is not None)
        result['proxy_width_reduction_pct'] = number(row.get('joint_estimated_width_reduction_pct'))
        result['valid_proxy'] = float(result['proxy_width_reduction_pct'] is not None)
    return result


def feature_catalog():
    result = []
    for name in SIGNALS:
        group = ('observed_bottle' if name.startswith('observed_') else 'hand_derived_proxy'
                 if name.startswith('proxy_') else 'hand_angles' if 'deg' in name else 'normalized_hand_geometry')
        unit = 'degrees' if 'deg' in name else 'percent' if 'pct' in name else 'dimensionless'
        result.append({'frame_feature': name, 'group': group, 'unit': unit,
                       'trial_summaries': ['p10', 'p50', 'p90', 'range'],
                       'range_definition': 'p90 minus p10 during squeeze',
                       'is_direct_force_measurement': False})
    return {'feature_version': 'grip_trial_v1', 'frame_features': result,
            'angles': '14 MediaPipe world-landmark angles; MCP uses wrist proxy',
            'flexion_change': 'relaxed_reference_angle minus current_angle',
            'geometry_normalization': 'lengths divided by palm width; hull area divided by palm width squared',
            'temporal_signal_names': [signal for signal, _ in EVENT_SIGNALS.values()],
            'temporal_features': {'release_residual': 'median during release, in signal units',
                                  'rise_speed_p90': '90th percentile positive slope, signal units/second; gaps over 0.3 seconds excluded',
                                  'time_to_peak_s': 'seconds from squeeze onset to first reaching 90% of squeeze p90',
                                  'hold_fraction_above_half': 'fraction of supported squeeze frames at least 50% of squeeze p90'},
            'target_semantics': 'one maximal dynamometer grip label per participant/trial, never per-frame force'}


def _percentile(values, q):
    values = [number(v) for v in values]
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else None


def summarize_rows(rows, recording_id='', source=''):
    # A frame contributes once, regardless of GUI refreshes or duplicate CSVs.
    unique, seen = [], set()
    for row in rows:
        stamp = number(row.get('time_s'))
        key = row.get('source_frame_id') or stamp
        if stamp is not None and key not in seen:
            seen.add(key)
            unique.append(row)
    rows = sorted(unique, key=lambda r: float(r['time_s']))
    if not rows:
        raise ValueError('No timestamped frames in recording')
    features = [row_features(r) for r in rows]
    times = np.array([float(r['time_s']) for r in rows])
    phases = [r.get('trial_phase', 'unknown') for r in rows]
    squeeze = np.array([p == 'squeeze' for p in phases])
    rest, release = np.array([p == 'rest' for p in phases]), np.array([p == 'release' for p in phases])
    guided = bool(np.any(squeeze) and np.any(rest) and np.any(release))
    selected = squeeze if np.any(squeeze) else np.ones(len(rows), bool)
    result = {}
    for name in SIGNALS:
        values = [f[name] for i, f in enumerate(features) if selected[i]]
        for suffix, q in (('p50', 50), ('p90', 90), ('p10', 10)):
            result[name + '_' + suffix] = _percentile(values, q)
        p90, p10 = result[name + '_p90'], result[name + '_p10']
        result[name + '_range'] = p90 - p10 if p90 is not None and p10 is not None else None
    events, visible_returns = [], []
    for family, (signal, minimum) in EVENT_SIGNALS.items():
        values = np.array([np.nan if f[signal] is None else f[signal] for f in features])
        peak = _percentile(values[selected], 90)
        result[signal + '_release_residual'] = _percentile(values[release], 50)
        speed = []
        for i in range(1, len(rows)):
            dt = times[i] - times[i - 1]
            if selected[i] and 0 < dt <= .3 and np.isfinite(values[i - 1:i + 1]).all():
                speed.append((values[i] - values[i - 1]) / dt)
        result[signal + '_rise_speed_p90'] = _percentile([v for v in speed if v > 0], 90)
        active_times = times[selected & np.isfinite(values)]
        active_values = values[selected & np.isfinite(values)]
        result[signal + '_time_to_peak_s'] = None
        result[signal + '_hold_fraction_above_half'] = None
        if peak is not None and peak > 0 and len(active_times):
            hits = active_times[active_values >= peak * .9]
            if len(hits):
                result[signal + '_time_to_peak_s'] = float(hits[0] - times[np.flatnonzero(selected)[0]])
            result[signal + '_hold_fraction_above_half'] = float(np.mean(active_values >= peak * .5))
        if guided and peak is not None:
            r = values[rest & np.isfinite(values)]
            end = values[release & np.isfinite(values)]
            active = values[squeeze & np.isfinite(values)]
            if len(r) >= 4 and len(end) >= 4 and len(active) >= 8:
                noise = float(np.median(np.abs(r - np.median(r)))) * 1.4826
                rise = peak - float(np.median(r))
                fall = peak - float(np.median(end))
                if rise >= max(minimum, 3 * noise):
                    events.append(family)
                    if fall >= max(minimum * .3, rise * .25):
                        visible_returns.append(family)
    labels = {number(r.get('reference_grip_kgf')) for r in rows if number(r.get('reference_grip_kgf')) is not None}
    if len(labels) > 1:
        raise ValueError('Reference label changes within a recording')
    def consistent(name, default=''):
        values = {r.get(name, '') for r in rows if r.get(name, '') != ''}
        if len(values) > 1:
            raise ValueError(f'{name} changes within a recording')
        return next(iter(values), default)
    metadata = {n: consistent(n) for n in ('participant_id', 'trial_id', 'bottle_id', 'tested_hand', 'label_source', 'trial_kind')}
    quality = {'total_frames': len(rows), 'squeeze_frames': int(np.count_nonzero(selected)),
               'duration_s': float(times[-1] - times[0]), 'guided_phases_present': guided,
               'event_families': events, 'visible_release_return_families': visible_returns,
               'hand_valid_fraction': float(np.mean([bool(f['valid_hand']) for f in features])),
               'observed_valid_fraction': float(np.mean([bool(f['valid_observed']) for f in features])),
               'proxy_valid_fraction': float(np.mean([bool(f['valid_proxy']) for f in features])),
               'axis_valid_fraction': float(np.mean([bool(f['valid_axis']) for f in features])),
               'display_only_fraction': float(np.mean([truth(r.get('joint_display_only')) for r in rows]))}
    valid_squeeze = sum(selected[i] and (bool(f['valid_hand']) or bool(f['valid_observed']) or bool(f['valid_proxy']))
                        for i, f in enumerate(features))
    issues = []
    if metadata['trial_kind'] != 'maximal':
        issues.append('not_declared_maximal_trial')
    if not guided:
        issues.append('missing_rest_squeeze_release_phases')
    if valid_squeeze < 8 or quality['duration_s'] < 4:
        issues.append('too_few_supported_frames')
    if quality['axis_valid_fraction'] < .6:
        issues.append('bottle_axis_missing_too_often')
    if not events:
        issues.append('no_supported_squeeze_release_cycle')
    # PET can remain plastically deformed after release. Record its residual
    # response; a visible elastic return is not a prerequisite for this study.
    warnings = ['no_visible_return_after_release'] if events and not visible_returns else []
    quality.update(usable_for_model=not issues, issues=issues, warnings=warnings)
    return {'schema_version': 1, 'feature_version': 'grip_trial_v1', 'recording_id': recording_id,
            'source_csv': str(source), **metadata, 'reference_grip_kgf': next(iter(labels), None),
            'label_is_frame_force': False, 'features': result, 'quality': quality}


def extract_recording(path):
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    result = summarize_rows(rows, path.stem, str(path.resolve()))
    context_path = path.with_suffix('.json')
    if context_path.is_file():
        context = json.loads(context_path.read_text(encoding='utf-8'))
        if context.get('grip_trial_guided') and not context.get('grip_trial_completed'):
            result['quality']['usable_for_model'] = False
            result['quality']['issues'].append('guided_trial_interrupted')
    result['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.features.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    flat = {k: result[k] for k in ('recording_id', 'participant_id', 'bottle_id', 'tested_hand', 'reference_grip_kgf', 'label_source')}
    flat.update(usable_for_model=result['quality']['usable_for_model'], **result['features'])
    with path.with_suffix('.features.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return result
