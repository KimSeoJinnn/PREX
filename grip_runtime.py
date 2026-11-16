"""Guided trials and local trial-level inference; no live training on labels."""

import hashlib
import json
import math
from pathlib import Path

from grip_features import extract_recording, number
from grip_model import calibrate_personal, predict_personal, predict_trial, save_json


ROOT = Path(__file__).resolve().parent


def load_settings(path=None):
    path = Path(path or ROOT / 'grip_settings.json')
    settings = json.loads(path.read_text(encoding='utf-8'))
    if number(settings.get('reference_grip_kgf')) is None or settings['reference_grip_kgf'] <= 0:
        raise ValueError('Settings reference_grip_kgf must be positive')
    for name in ('guided_rest_s', 'guided_squeeze_s', 'guided_release_s'):
        value = number(settings.get(name))
        if value is None or not .5 <= value <= 15:
            raise ValueError(f'{name} must be between .5 and 15 seconds')
    return settings


def label_metadata(settings, participant=None, grip=None, label_source=None, bottle=None,
                   tested_hand=None, predict_only=False):
    participant = participant or settings['participant_id']
    is_default_person = participant == settings['participant_id']
    value = (grip if grip is not None else settings['reference_grip_kgf'] if is_default_person else None)
    source = label_source or ('dynamometer' if grip is not None else settings['label_source'] if is_default_person else 'unspecified')
    if predict_only:
        value, source = None, 'unspecified'
    return {'participant_id': participant, 'reference_grip_kgf': value if value is not None else '',
            'label_source': source, 'bottle_id': bottle or settings['bottle_id'],
            'tested_hand': tested_hand or settings['tested_hand']}


class GripExperiment:
    def __init__(self, metadata, settings=None, root=None, model_path=None):
        self.root = Path(root or ROOT)
        self.settings = settings or load_settings()
        self.metadata = metadata
        self.started = None
        self.guided = False
        self.reference_consumed = False
        self.completed_guide = False
        self.last_result = None
        self.last_record = None
        self.message = 'H: baseline, G: guided maximal trial'
        self.model_path = Path(model_path or self.root / 'models/grip_estimator.json')
        self.population = self._load(self.model_path)

    @staticmethod
    def _load(path):
        if not Path(path).is_file():
            return None
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def reset_reference(self):
        self.reference_consumed = False

    @property
    def duration(self):
        return sum(self.settings[k] for k in ('guided_rest_s', 'guided_squeeze_s', 'guided_release_s'))

    def begin(self, now, guided=False):
        self.started, self.guided = now, guided
        self.completed_guide = False
        self.last_result = None

    def phase(self, now):
        if self.started is None or not self.guided:
            return 'unknown'
        elapsed = now - self.started
        a, b = self.settings['guided_rest_s'], self.settings['guided_squeeze_s']
        return 'rest' if elapsed < a else 'squeeze' if elapsed < a + b else 'release'

    def instruction(self, now):
        if self.started is None or not self.guided:
            return self.message
        elapsed = now - self.started
        a, b = self.settings['guided_rest_s'], self.settings['guided_squeeze_s']
        phase = self.phase(now)
        end = a if phase == 'rest' else a + b if phase == 'squeeze' else self.duration
        text = {'rest': 'RELAX grip', 'squeeze': 'SQUEEZE to your maximum',
                'release': 'RELAX; keep holding bottle lightly'}[phase]
        return f'{text}  {max(0, math.ceil(end - elapsed))}s'

    def due(self, now):
        return self.guided and self.started is not None and now - self.started >= self.duration

    def personal_path(self, record):
        identity = '|'.join(str(record.get(k, '')) for k in ('participant_id', 'bottle_id', 'tested_hand'))
        key = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return self.root / 'models/personal_grip' / f'{key}.json'

    def finish(self, csv_path, completed=False):
        guided = self.guided
        self.completed_guide = bool(guided and completed)
        self.started, self.guided = None, False
        if guided:
            self.reference_consumed = True
        try:
            record = extract_recording(csv_path)
            self.last_record = record
            population_result = predict_trial(self.population, record) if self.population is not None else None
            if guided and not completed:
                self.message = 'Trial interrupted; features saved, repeat H then G'
                self.last_result = {'status': 'interrupted_trial', 'kgf': None}
            elif not record['quality']['usable_for_model']:
                self.message = 'Features saved; need a clear squeeze-release cycle'
                self.last_result = {'status': 'insufficient_trial_quality', 'kgf': None,
                                    'issues': record['quality']['issues']}
            elif population_result is not None and population_result.get('kgf') is not None:
                self.last_result = population_result
                self.message = 'Research model result saved; H for next trial'
            else:
                path = self.personal_path(record)
                artifact = self._load(path)
                upgrade = (artifact is not None and artifact.get('label_source') != 'dynamometer'
                           and record.get('label_source') == 'dynamometer')
                if upgrade:
                    artifact = calibrate_personal(record)
                    save_json(path, artifact)
                    self.last_result = {'status': 'personal_calibration_saved',
                                        'reference_kgf': record['reference_grip_kgf'], 'kgf': None,
                                        'label_source': 'dynamometer', 'accuracy_validated': False}
                    self.message = 'Measured reference replaced temporary label; H for next test'
                elif artifact is not None:
                    self.last_result = predict_personal(artifact, record)
                    self.message = 'Personal scaling result saved; H for next trial'
                elif (self.settings.get('personal_auto_calibration') and record.get('participant_id')
                      and number(record.get('reference_grip_kgf')) is not None):
                    artifact = calibrate_personal(record)
                    save_json(path, artifact)
                    self.last_result = {'status': 'personal_calibration_saved',
                                        'reference_kgf': record['reference_grip_kgf'], 'kgf': None,
                                        'label_source': record.get('label_source'), 'accuracy_validated': False}
                    self.message = 'Personal reference saved; H then G for a new test'
                else:
                    self.last_result = {'status': 'need_model_or_personal_reference', 'kgf': None}
                    self.message = 'Features saved; need labeled calibration or trained model'
                if population_result is not None:
                    self.last_result['population_model_status'] = population_result['status']
            save_json(Path(csv_path).with_suffix('.estimate.json'), self.last_result)
            print('Grip trial:', json.dumps(self.last_result, ensure_ascii=False), flush=True)
        except (OSError, ValueError, KeyError) as error:
            self.message = 'Feature export failed; raw CSV retained'
            self.last_result = {'status': 'feature_export_error', 'kgf': None, 'error': str(error)}
            print(f'Grip feature export error: {error}', flush=True)
        return self.last_result
