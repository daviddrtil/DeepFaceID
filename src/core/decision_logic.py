from passive import passive_runner

class DecisionLogic:
    def fuse(self, interactive_actions, passive_score_avg, current_action, actions_completed_count, actions_count, timeout_failed=False):
        if timeout_failed:
            return {
                'status': 'fail',
                'display_status': 'Action Timeout',
                'passive_ok': passive_score_avg is not None and passive_score_avg <= passive_runner.DEEPFAKE_SCORE_THRESHOLD,
                'interactive_complete': False,
            }

        passive_ok = passive_score_avg is not None and passive_score_avg <= passive_runner.DEEPFAKE_SCORE_THRESHOLD
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
        }
