class DecisionLogic:
    DEEPFAKE_SCORE_THRESHOLD = 0.50     # Average passive score above this indicates likely deepfake
    SIMILARITY_REJECT_THRESHOLD = 0.20  # Cosine similarity below this triggers immediate rejection
    IDENTITY_SCORE_THRESHOLD = 0.50     # Minimum identity score required to pass final decision

    def fuse(self, passive_result, identity_result, actions_completed_count, actions_count, timeout_failed=False):
        passive_score_avg = passive_result.score_avg if passive_result else None
        passive_ok = passive_score_avg is not None and passive_score_avg <= self.DEEPFAKE_SCORE_THRESHOLD

        identity_ok = True
        identity_rejection = False
        if identity_result is not None and identity_result.embedding_count >= 10:
            identity_ok = identity_result.identity_score >= self.IDENTITY_SCORE_THRESHOLD
            identity_rejection = identity_result.similarity is not None and identity_result.similarity < self.SIMILARITY_REJECT_THRESHOLD

        if timeout_failed:
            return {
                'status': 'fail',
                'display_status': 'Action Timeout',
                'passive_ok': passive_ok,
                'identity_ok': identity_ok,
                'interactive_complete': False,
                'passive': passive_result,
            }

        if identity_rejection:
            return {
                'status': 'fail',
                'display_status': 'Identity Mismatch',
                'passive_ok': passive_ok,
                'identity_ok': False,
                'interactive_complete': False,
                'passive': passive_result,
            }

        interactive_complete = actions_count > 0 and actions_completed_count >= actions_count
        if passive_ok and identity_ok and interactive_complete:
            status = 'pass'
            display_status = 'Authorized'
        elif interactive_complete:
            status = 'fail'
            if not identity_ok:
                display_status = 'Identity Inconsistent'
            else:
                display_status = 'Failed'
        else:
            status = 'pending'
            display_status = f'{actions_completed_count}/{actions_count} actions completed'

        return {
            'status': status,
            'display_status': display_status,
            'passive_ok': passive_ok,
            'identity_ok': identity_ok,
            'interactive_complete': interactive_complete,
            'passive': passive_result,
        }
