from interactive.action_enum import get_action_category


ACTION_WEIGHTS = {
    'complex': 2.0,
    'occlusion': 2.0,
    'pose': 1.2,
    'sequence': 1.0,
    'expression': 0.8,
    'calibration': 0.5,
}


class DecisionLogic:
    DEEPFAKE_SCORE_THRESHOLD = 0.25
    SIMILARITY_REJECT_THRESHOLD = 0.20
    IDENTITY_SCORE_THRESHOLD = 0.45
    CONFIDENT_FAKE_SCORE = 0.90

    def __init__(self):
        self._action_scores = []
        self._action_start_frame = 0
        self._deepfake_flagged = False

    def complete_action(self, completed_action, frame_count, passive_runner):
        category = get_action_category(completed_action)
        weight = ACTION_WEIGHTS.get(category, 1.0)
        start = self._action_start_frame
        spatial_buf = passive_runner.spatial.get_score_buffer()
        temporal_buf = passive_runner.temporal.get_score_buffer()

        spatial_scores = [v for f, v in spatial_buf.items() if start <= f <= frame_count]
        temporal_scores = [v for f, v in temporal_buf.items() if start <= f <= frame_count]

        spatial_max = max(spatial_scores) if spatial_scores else 0.0
        temporal_max = max(temporal_scores) if temporal_scores else 0.0

        action_score = spatial_max * 0.7 + temporal_max * 0.3
        self._action_scores.append((action_score, weight))
        self._action_start_frame = frame_count

    def _compute_deepfake_score(self, passive_result, identity_result, passive_runner):
        weighted_action = 0.0
        if self._action_scores:
            total_w = sum(w for _, w in self._action_scores)
            weighted_action = sum(s * w for s, w in self._action_scores) / total_w if total_w else 0.0

        spatial_max = passive_result.spatial.max_score if passive_result and passive_result.spatial.max_score else 0.0
        temporal_max = passive_result.temporal.max_score if passive_result and passive_result.temporal.max_score else 0.0
        passive_avg = passive_result.score_avg if passive_result and passive_result.score_avg else 0.0

        # Count temporal frames above fake threshold
        temporal_buf = passive_runner.temporal.get_score_buffer() if passive_runner else {}
        temporal_fake_ratio = 0.0
        if temporal_buf:
            above = sum(1 for v in temporal_buf.values() if v > 0.5)
            temporal_fake_ratio = above / len(temporal_buf)

        # Identity penalty: low similarity = suspicious
        identity_penalty = 0.0
        if identity_result and identity_result.embedding_count >= 10:
            id_score = identity_result.identity_score or 0.0
            identity_penalty = max(0.0, 0.5 - id_score)

        # Weighted composite
        signals = {
            'action': (weighted_action, 3.0),
            'spatial_max': (spatial_max, 2.0),
            'temporal_max': (temporal_max, 1.0),
            'temporal_fake_ratio': (temporal_fake_ratio, 1.0),
            'passive_avg': (passive_avg, 0.5),
            'identity_penalty': (identity_penalty, 1.5),
        }

        total_weight = sum(w for _, w in signals.values())
        score = sum(v * w for v, w in signals.values()) / total_weight if total_weight else 0.0
        return min(1.0, score)

    def fuse(self, passive_result, identity_result, actions_completed_count, actions_count, timeout_failed=False, passive_runner=None):
        deepfake_score = self._compute_deepfake_score(passive_result, identity_result, passive_runner)

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
        }

        if timeout_failed:
            return {**base, 'status': 'fail', 'display_status': 'Action Timeout', 'interactive_complete': False}

        if identity_rejection:
            return {**base, 'status': 'fail', 'display_status': 'Identity Mismatch', 'identity_ok': False, 'interactive_complete': False}

        interactive_complete = actions_count > 0 and actions_completed_count >= actions_count

        if not interactive_complete:
            return {**base, 'status': 'pending', 'display_status': f'{actions_completed_count}/{actions_count} actions completed', 'interactive_complete': False}

        if passive_ok and identity_ok:
            return {**base, 'status': 'pass', 'display_status': 'Authorized', 'interactive_complete': True}

        display = 'Identity Inconsistent' if not identity_ok else 'Failed'
        return {**base, 'status': 'fail', 'display_status': display, 'interactive_complete': True}
