import csv
import json
from pathlib import Path

import settings
from interactive.action_enum import get_action_name, get_action_category

_FRAME_COLUMNS = [
    'frame',
    'spatial_frame', 'frequency_frame', 'temporal_frame',
    'deepfake_score', 'decision',
    'passive_cur', 'passive_avg', 'passive_max', 'passive_smooth',
    'spatial', 'spatial_avg', 'spatial_max',
    'frequency', 'frequency_avg', 'frequency_max',
    'temporal', 'temporal_avg', 'temporal_max',
    'id_sim', 'id_avg', 'id_min', 'id_drift', 'id_score',
    'face', 'hand', 'overlap',
    'yaw', 'pitch', 'roll',
    'challenge_index', 'challenge_progress',
    'action_category', 'action',
    'pose', 'occlusions', 'expressions',
]


class StatisticsWriter:
    STATS_FILE   = 'stats.csv'
    SUMMARY_FILE = 'summary.csv'

    def __init__(self):
        out = Path(settings.config.output_dir)
        self._frames_file = open(out / self.STATS_FILE, 'w', encoding='utf-8', newline='')
        self._writer = csv.DictWriter(self._frames_file, fieldnames=_FRAME_COLUMNS)
        self._writer.writeheader()
        self._frames_file.flush()
        self._summary_path = out / self.SUMMARY_FILE

    @staticmethod
    def _f(value):
        return round(float(value), 4) if value is not None else None

    @staticmethod
    def _action_list(values):
        return '|'.join(v.name for v in (values or []) if v is not None)

    def write_frame(self, frame_count, interactive_result, passive_result, identity_result, current_action, challenge_index, challenge_total, decision_result=None):
        actions = interactive_result.actions
        record = {
            'frame': frame_count,
            'spatial_frame': 0, 'frequency_frame': 0, 'temporal_frame': 0,
            'passive_cur': None, 'passive_avg': None, 'passive_max': None, 'passive_smooth': None,
            'spatial': None, 'spatial_avg': None, 'spatial_max': None,
            'frequency': None, 'frequency_avg': None, 'frequency_max': None,
            'temporal': None, 'temporal_avg': None, 'temporal_max': None,
        }
        if passive_result is not None:
            record.update({
                'passive_cur':    self._f(passive_result.score_cur),
                'passive_avg':    self._f(passive_result.score_avg),
                'passive_max':    self._f(passive_result.score_max),
                'passive_smooth': self._f(passive_result.score_smooth),
                'spatial':     self._f(passive_result.spatial.current_score),
                'spatial_avg': self._f(passive_result.spatial.avg_score),
                'spatial_max': self._f(passive_result.spatial.max_score),
                'frequency':     self._f(passive_result.frequency.current_score),
                'frequency_avg': self._f(passive_result.frequency.avg_score),
                'frequency_max': self._f(passive_result.frequency.max_score),
                'temporal':     self._f(passive_result.temporal.current_score),
                'temporal_avg': self._f(passive_result.temporal.avg_score),
                'temporal_max': self._f(passive_result.temporal.max_score),
                'spatial_frame':   passive_result.spatial.current_frame,
                'frequency_frame': passive_result.frequency.current_frame,
                'temporal_frame':  passive_result.temporal.current_frame,
            })
        record.update({
            'id_sim':   self._f(identity_result.similarity       if identity_result else None),
            'id_avg':   self._f(identity_result.avg_similarity   if identity_result else None),
            'id_min':   self._f(identity_result.min_similarity   if identity_result else None),
            'id_drift': self._f(identity_result.drift            if identity_result else None),
            'id_score': self._f(identity_result.identity_score   if identity_result else None),
            'face':    bool(actions.get('face_detected',    False)),
            'hand':    bool(actions.get('hand_detected',    False)),
            'overlap': bool(actions.get('hand_face_overlap', False)),
            'yaw':   self._f(actions.get('yaw')),
            'pitch': self._f(actions.get('pitch')),
            'roll':  self._f(actions.get('roll')),
            'challenge_index':    challenge_index,
            'challenge_progress': self._f(interactive_result.challenge_progress),
            'action_category': get_action_category(current_action) or None,
            'action':          get_action_name(current_action)     or None,
            'pose':        self._action_list(actions.get('pose')),
            'occlusions':  self._action_list(actions.get('occlusions')),
            'expressions': self._action_list(actions.get('expressions')),
            'deepfake_score': self._f(decision_result.get('deepfake_score') if decision_result else None),
            'decision':       decision_result.get('status') if decision_result else None,
        })
        self._writer.writerow(record)
        self._frames_file.flush()

    @staticmethod
    def _build_summary(passive_result, identity_result, final_decision, deepfake_label, deepfake_score, temporal_window_stats):
        r = {}
        if passive_result:
            for key, res in [('spatial', passive_result.spatial), ('frequency', passive_result.frequency), ('temporal', passive_result.temporal)]:
                r[f'{key}_avg']   = round(res.avg_score, 4) if res.avg_score is not None else None
                r[f'{key}_max']   = round(res.max_score, 4) if res.max_score is not None else None
                r[f'{key}_count'] = res.total_count
        if temporal_window_stats is not None:
            max_w, fake_w, total_w = temporal_window_stats
            r.update({'temporal_max_window': round(max_w, 4), 'temporal_fake_windows': fake_w, 'temporal_total_windows': total_w})
        if identity_result:
            r.update({
                'identity_avg_sim':    round(identity_result.avg_similarity, 4),
                'identity_min_sim':    round(identity_result.min_similarity, 4),
                'identity_drift':      round(identity_result.drift,          4),
                'identity_score':      round(identity_result.identity_score, 4),
                'identity_embeddings': identity_result.embedding_count,
            })
        if deepfake_score is not None:
            r['deepfake_score'] = round(deepfake_score, 4)
        r['label'] = deepfake_label or 'unknown'
        r['final_decision'] = final_decision or 'unknown'
        return r

    def write_summary(self, passive_result, identity_result, final_decision, deepfake_label, deepfake_score=None, temporal_window_stats=None, decision_signals=None, challenge_total=None, challenge_sequence=None, passive_ok=None, identity_ok=None):
        record = self._build_summary(passive_result, identity_result, final_decision, deepfake_label, deepfake_score, temporal_window_stats)
        if challenge_total is not None:
            record['challenge_total'] = challenge_total
        if challenge_sequence is not None:
            record['challenge_sequence'] = challenge_sequence
        if passive_ok is not None:
            record['passive_ok'] = passive_ok
        if identity_ok is not None:
            record['identity_ok'] = identity_ok
        if decision_signals:
            for k, v in decision_signals.items():
                record[f'signal_{k}'] = v
        with open(self._summary_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(record.keys()))
            w.writeheader()
            w.writerow(record)
        return json.dumps(record)  # for console print in liveness_detection_engine

    def close(self):
        if not self._frames_file.closed:
            self._frames_file.close()
