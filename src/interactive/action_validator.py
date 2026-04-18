from interactive.action_enum import ComplexAction, SequenceAction, get_action_name


class ActionValidator:
    def validate(self, current_action, actions, challenge_timer, timestamp_ms):
        matched = self.is_match(current_action, actions)
        completed, progress = challenge_timer.update(matched, timestamp_ms)
        completed_value = get_action_name(current_action) if completed else None
        return completed_value, progress

    def is_match(self, current_action, actions):
        if not current_action:
            return False

        detected = set(actions.get('pose', []) + actions.get('expressions', []) + actions.get('occlusions', []))

        if isinstance(current_action, ComplexAction):
            return current_action.actions.issubset(detected)

        if isinstance(current_action, SequenceAction):
            return current_action.actions[0] in detected

        return current_action in detected
