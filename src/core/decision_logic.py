class DecisionLogic:
    DEEPFAKE_SCORE_THRESHOLD = 0.50

    def fuse(self, passive_result, actions_completed_count, actions_count, timeout_failed=False):
        passive_score_avg = passive_result.score_avg if passive_result else None
        passive_ok = passive_score_avg is not None and passive_score_avg <= self.DEEPFAKE_SCORE_THRESHOLD

        if timeout_failed:
            return {
                'status': 'fail',
                'display_status': 'Action Timeout',
                'passive_ok': passive_ok,
                'interactive_complete': False,
                'passive': passive_result,
            }

        interactive_complete = actions_count > 0 and actions_completed_count >= actions_count
        if passive_ok and interactive_complete:
            status = 'pass'
            display_status = 'Authorized'
        elif interactive_complete:
            status = 'fail'
            display_status = 'Failed'
        else:
            status = 'pending'
            display_status = f'{actions_completed_count}/{actions_count} actions completed'

        return {
            'status': status,
            'display_status': display_status,
            'passive_ok': passive_ok,
            'interactive_complete': interactive_complete,
            'passive': passive_result,
        }
