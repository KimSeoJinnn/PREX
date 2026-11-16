"""Offline extraction, correlation, nested model comparison and prediction.

Examples:
  python grip_analysis.py extract recordings
  python grip_analysis.py analyze --recordings recordings
  python grip_analysis.py predict --model models/grip_estimator.json trial.features.json
"""

import argparse
import csv
import html
import json
from pathlib import Path

from grip_features import extract_recording, feature_catalog
from grip_model import (correlations, prepare_subjects, train_evaluate, predict_trial,
                        predict_personal, calibrate_personal, save_json)


ROOT = Path(__file__).resolve().parent


def recording_paths(paths):
    result = []
    for path in paths:
        path = Path(path)
        candidates = sorted(path.glob('*.csv')) if path.is_dir() else [path]
        result.extend(p for p in candidates if p.suffix == '.csv' and '.features' not in p.stem)
    return sorted(set(p.resolve() for p in result))


def load_record(path):
    path = Path(path)
    return extract_recording(path) if path.suffix == '.csv' else json.loads(path.read_text(encoding='utf-8'))


def _html_report(report):
    escaped = lambda value: html.escape(str(value))
    rows = ''.join('<tr>' + ''.join(f'<td>{escaped(row.get(key, ""))}</td>' for key in
                                  ('feature', 'n_participants', 'pearson_r', 'spearman_r', 'fdr_q', 'reason')) + '</tr>'
                   for row in sorted(report['correlations'], key=lambda r: -abs(r['spearman_r'] or 0.))[:80])
    predictions = report['model_comparison'].get('oof_predictions', [])
    chart = ''
    if predictions:
        maximum = max(max(p['truth_kgf'], p['prediction_kgf']) for p in predictions) * 1.1
        dots = ''.join(f'<circle cx="{50 + p["truth_kgf"] / maximum * 360:.2f}" '
                       f'cy="{410 - p["prediction_kgf"] / maximum * 360:.2f}" r="5" fill="#1c806b">'
                       f'<title>{escaped(p["participant_id"])}: truth {p["truth_kgf"]:.1f}, '
                       f'prediction {p["prediction_kgf"]:.1f}</title></circle>' for p in predictions)
        chart = f'<svg viewBox="0 0 460 460" width="460"><path d="M50 50V410H410M50 410L410 50" fill="none" stroke="#aaa"/>{dots}<text x="155" y="450">실제 악력 kgf</text><text x="55" y="30">참가자 분리 검증 예측 kgf</text></svg>'
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>악력 feature 분석</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#21302c}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}pre{{white-space:pre-wrap;background:#f3f6f4;padding:18px}}.notice{{background:#fff3d9;padding:18px}}</style>
<h1>영상 기반 악력 추정 실험</h1><p class="notice">80kgf는 본인용 임시 정답입니다. 한 사람의 정답 하나에 맞춘 계산은 정확도 검증이 아닙니다. 정답은 최대 악력의 trial 라벨이며 프레임별 힘이 아닙니다.</p>
<p>녹화 {report['recordings']}개 · 분석 대상 참가자 {report['model_comparison']['n_participants']}명 · 상태: {escaped(report['model_comparison']['status'])}</p>
<h2>모델 비교</h2>{chart}<pre>{escaped(json.dumps(report['model_comparison'], ensure_ascii=False, indent=2))}</pre>
<h2>Feature와 악력의 상관관계</h2><p>동일인의 반복 영상을 합쳐 참가자별로 계산합니다. 결측값·동일한 정답값은 상관계수를 만들지 않습니다. 여러 검정은 FDR로 보정합니다.</p>
<table><tr><th>Feature</th><th>참가자 수</th><th>Pearson</th><th>Spearman</th><th>FDR q</th><th>계산 불가 이유</th></tr>{rows}</table>
<h2>데이터 제외 사유</h2><pre>{escaped(json.dumps(report['excluded'], ensure_ascii=False, indent=2))}</pre>
<p>평가 설계 참고: <a href="https://arxiv.org/abs/1606.05201">중첩 교차검증의 한계와 방법</a>. 임상적 정확도는 검증되지 않았습니다.</p></html>'''


def analyze(records, output_dir, include_provisional=False, bottle_id=None, tested_hand=None, model_path=None):
    if tested_hand:
        records = [r for r in records if r.get('tested_hand') == tested_hand]
    subjects, excluded = prepare_subjects(records, include_provisional, bottle_id)
    comparison, artifact = train_evaluate(subjects)
    corr = correlations(subjects if comparison['status'] != 'mixed_conditions' else [])
    if comparison['status'] == 'mixed_conditions':
        for row in corr:
            row['reason'] = 'mixed_bottle_or_hand_conditions_select_one'
    report = {'schema_version': 1, 'recordings': len(records),
              'include_provisional_labels': include_provisional,
              'unit_of_analysis': 'one_median_trial_summary_per_participant_and_condition',
              'frame_rows_are_not_independent_subjects': True,
              'model_comparison': comparison, 'correlations': corr, 'excluded': excluded,
              'records': [{'recording_id': r['recording_id'], 'reference_grip_kgf': r['reference_grip_kgf'],
                           'label_source': r['label_source'], 'quality': r['quality']} for r in records]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / 'grip_report.json', report)
    save_json(output_dir / 'trial_dataset.json', records)
    save_json(output_dir / 'grip_feature_catalog.json', feature_catalog())
    with (output_dir / 'correlations.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['feature', 'n_participants', 'pearson_r', 'spearman_r', 'permutation_p', 'fdr_q', 'reason'])
        writer.writeheader()
        writer.writerows(corr)
    (output_dir / 'grip_report.html').write_text(_html_report(report), encoding='utf-8')
    if artifact is not None and model_path is not None:
        save_json(model_path, artifact)
    return report, artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local grip features and participant-separated learning; no paid services')
    commands = parser.add_subparsers(dest='command', required=True)
    extract = commands.add_parser('extract', help='export trial features without assigning new labels')
    extract.add_argument('paths', nargs='+', type=Path)
    analysis = commands.add_parser('analyze', help='correlations and nested model comparison')
    analysis.add_argument('--recordings', type=Path, default=ROOT / 'recordings')
    analysis.add_argument('--output', type=Path, default=ROOT / 'analysis')
    analysis.add_argument('--model-output', type=Path, default=ROOT / 'models/grip_estimator.json')
    analysis.add_argument('--include-provisional', action='store_true', help='exploratory only: include assumed labels')
    analysis.add_argument('--bottle-id')
    analysis.add_argument('--tested-hand')
    prediction = commands.add_parser('predict', help='predict from features, without using this trial ground-truth label')
    prediction.add_argument('recording', type=Path)
    prediction.add_argument('--model', required=True, type=Path)
    calibration = commands.add_parser('calibrate-personal', help='explicit one-point demonstration, not validated force learning')
    calibration.add_argument('recording', type=Path)
    calibration.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in ('extract', 'analyze'):
            paths = recording_paths(args.paths if args.command == 'extract' else [args.recordings])
            records, errors = [], []
            for path in paths:
                try:
                    records.append(extract_recording(path))
                except (OSError, ValueError) as error:
                    errors.append({'file': str(path), 'error': str(error)})
            if args.command == 'extract':
                print(json.dumps({'recordings_extracted': len(records), 'errors': errors}, ensure_ascii=False, indent=2))
            else:
                report, artifact = analyze(records, args.output, args.include_provisional, args.bottle_id,
                                           args.tested_hand, args.model_output)
                print(json.dumps({'report': str(args.output / 'grip_report.html'),
                                  'model_saved': artifact is not None, 'status': report['model_comparison']['status'],
                                  'n_participants': report['model_comparison']['n_participants'], 'read_errors': errors},
                                 ensure_ascii=False, indent=2))
        elif args.command == 'calibrate-personal':
            save_json(args.output, calibrate_personal(load_record(args.recording)))
            print('Personal scaling reference saved; accuracy is not validated.')
        else:
            artifact = json.loads(args.model.read_text(encoding='utf-8'))
            record = load_record(args.recording)
            result = (predict_personal(artifact, record) if artifact.get('kind') == 'personal_reference_scaling_demo'
                      else predict_trial(artifact, record))
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'Analysis error: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
