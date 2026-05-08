import math
from interactive.action_enum import get_action_category


ACTION_WEIGHTS = {
    # Higher weights are assigned to actions expected to expose more artifacts, but kept moderate to limit false positives
    'complex': 1.25,
    'occlusion': 1.2,
    'pose': 1.2,
    'sequence': 1.0,
    'expression': 0.8,
    'calibration': 0.5,
}


def _sigmoid(x, center, steepness):
    # Calibration sigmoid: maps a raw detector score to evidence strength [0, 1]
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def _scores_in_range(buffer, start_frame, end_frame):
    return [v for f, v in sorted(buffer.items()) if start_frame <= f <= end_frame]


class DecisionLogic:
    DEEPFAKE_SCORE_THRESHOLD = 0.50
    SIMILARITY_REJECT_THRESHOLD = 0.20
    IDENTITY_SCORE_THRESHOLD = 0.45
    CONFIDENT_FAKE_SCORE = 0.90

    def __init__(self):
        self._action_scores = []  # (deepfake_score, weight) per completed action
        self._deepfake_flagged = False

    def score_action(self, completed_action, frame_count, passive_runner, hold_duration_frames=45):
        category = get_action_category(completed_action)
        weight = ACTION_WEIGHTS.get(category, 1.0)

        start = max(0, frame_count - hold_duration_frames + 1)
        spatial_scores = _scores_in_range(passive_runner.spatial.get_score_buffer(), start, frame_count)
        temporal_scores = _scores_in_range(passive_runner.temporal.get_score_buffer(), start, frame_count)

        # intentionally excluding indentity_penalty because it is at session-level
        deepfake_score, signals = self._compute_deepfake_score(spatial_scores, temporal_scores)
        self._action_scores.append((deepfake_score, weight))

        spatial_max = max(spatial_scores) if spatial_scores else 0.0
        spatial_avg = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 0.0
        temporal_max = max(temporal_scores) if temporal_scores else 0.0
        temporal_avg = sum(temporal_scores) / len(temporal_scores) if temporal_scores else 0.0

        return {
            'frame_start': start,
            'frame_end': frame_count,
            'frame_count': frame_count - start + 1,
            'spatial_max': round(spatial_max,  4),
            'spatial_avg': round(spatial_avg,  4),
            'spatial_samples': len(spatial_scores),
            'temporal_max': round(temporal_max, 4),
            'temporal_avg': round(temporal_avg, 4),
            'temporal_samples': len(temporal_scores),
            'deepfake_score': round(deepfake_score, 4),
            'weight': weight,
            'signals': signals,
        }

    def _compute_deepfake_score(self, spatial_scores, temporal_scores, weighted_action=None, identity_penalty=None):
        spatial_max = max(spatial_scores) if spatial_scores else 0.0
        spatial_avg = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 0.0
        temporal_max = max(temporal_scores) if temporal_scores else 0.0

        # Calibrate detector peaks to evidence strength
        spatial_evidence = _sigmoid(spatial_max, center=0.75, steepness=9.2)     # above 0.9 is score > 0.5,  more gradual
        temporal_evidence = _sigmoid(temporal_max, center=0.90, steepness=28.0)  # above 0.95 is score > 0.5, more strict

        signals = {
            'spatial_peak': (spatial_evidence, 2.5),
            'spatial_avg': (min(1.0, spatial_avg * 5), 1.0),  # amplified avg score based on calibration
        }
        if temporal_scores:
            signals['temporal_peak'] = (temporal_evidence, 1.0)
        if weighted_action is not None:
            signals['weighted_signal_action'] = (weighted_action, 1.0)
        if identity_penalty is not None:
            signals['identity_penalty'] = (identity_penalty, 1.0)

        total_weight = sum(w for _, w in signals.values())
        avg_score = sum(v * w for v, w in signals.values()) / total_weight if total_weight else 0.0

        # Let strong peak evidence override average score
        max_evidence = max(spatial_evidence, temporal_evidence)
        boost_alpha = _sigmoid(max_evidence, center=0.775, steepness=17.6)
        score = (1.0 - boost_alpha) * avg_score + boost_alpha * max_evidence

        signal_values = {
            'weighted_signal_action': round(weighted_action, 4) if weighted_action is not None else 0.0,
            'spatial_evidence': round(spatial_evidence, 4),
            'temporal_evidence': round(temporal_evidence, 4),
            'identity_penalty': round(identity_penalty, 4) if identity_penalty is not None else 0.0,
            'avg_score': round(avg_score, 4),
            'boost_alpha': round(boost_alpha, 4),
        }
        return min(1.0, score), signal_values

    def _weighted_action_score(self):
        if not self._action_scores:
            return None
        total_w = sum(w for _, w in self._action_scores)
        if not total_w:
            return 0.0
        return sum(s * w for s, w in self._action_scores) / total_w

    def _identity_penalty(self, identity_result):
        if identity_result is None or identity_result.embedding_count < 10:
            return 0.0
        return max(0.0, 0.70 - (identity_result.identity_score or 0.0))

    def fuse(self, passive_result, identity_result, actions_completed_count, actions_count, timeout_failed=False, passive_runner=None):
        spatial_scores = list(passive_runner.spatial.get_score_buffer().values()) if passive_runner else []
        temporal_scores = list(passive_runner.temporal.get_score_buffer().values()) if passive_runner else []

        deepfake_score, signals = self._compute_deepfake_score(
            spatial_scores,
            temporal_scores,
            weighted_action=self._weighted_action_score(),
            identity_penalty=self._identity_penalty(identity_result),
        )

        if deepfake_score >= self.CONFIDENT_FAKE_SCORE:
            self._deepfake_flagged = True

        passive_ok = deepfake_score <= self.DEEPFAKE_SCORE_THRESHOLD and not self._deepfake_flagged

        identity_ok = True
        identity_rejection = False
        if identity_result is not None and identity_result.embedding_count >= 10:
            identity_ok = identity_result.identity_score >= self.IDENTITY_SCORE_THRESHOLD
            identity_rejection = identity_result.similarity is not None and identity_result.similarity < self.SIMILARITY_REJECT_THRESHOLD

        base = {
            'passive_ok': passive_ok,
            'identity_ok': identity_ok,
            'passive': passive_result,
            'deepfake_score': deepfake_score,
            'signals': signals,
            'deepfake_flagged': self._deepfake_flagged,
        }

        if timeout_failed:
            return {**base, 'status': 'fail', 'display_status': 'Action Timeout', 'interactive_complete': False}

        # Re-enable after system testing and experimenting, where we dont want to auto-reject
        # if identity_rejection:
        #     return {**base, 'status': 'fail', 'display_status': 'Identity Mismatch', 'identity_ok': False, 'interactive_complete': False}

        interactive_complete = actions_count > 0 and actions_completed_count >= actions_count

        if not interactive_complete:
            return {**base, 'status': 'pending', 'display_status': f'{actions_completed_count}/{actions_count} actions completed', 'interactive_complete': False}

        if passive_ok and identity_ok:
            return {**base, 'status': 'pass', 'display_status': 'Authorized', 'interactive_complete': True}

        # Display_status is 'Identity Inconsistent' if not identity_ok else 'Not Authorized'
        display_status = 'Not Authorized'
        return {**base, 'status': 'fail', 'display_status': display_status, 'interactive_complete': True}
