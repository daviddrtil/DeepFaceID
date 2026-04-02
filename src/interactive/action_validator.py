class ActionValidator:
    def validate(self, current_action, actions, challenge_timer, timestamp_ms):
        matched = self.is_match(current_action, actions)
        completed, progress = challenge_timer.update(matched, timestamp_ms)
        return current_action.value if (completed and current_action) else None, progress

    def is_match(self, current_action, actions):
        if not current_action:
            return False

        return (current_action in actions.get('pose', [])
                or current_action in actions.get('expressions', [])
                or current_action in actions.get('occlusions', []))
