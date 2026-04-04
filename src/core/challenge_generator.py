import random
from interactive.action_enum import PoseType, OcclusionType, MovementType, ChallengeType, COMPLEX_ACTIONS, ACTION_SEQUENCES


class ChallengeGenerator:
    def __init__(self, actions_count=6):
        self.actions: list[ChallengeType] = self._generate_actions(actions_count)
        self.current_index = 0

    def _generate_actions(self, count):
        sequence = random.choice(ACTION_SEQUENCES)
        actions: list[ChallengeType] = [
            random.choice(list(OcclusionType)),
            random.choice(COMPLEX_ACTIONS),
            *random.sample(sequence.actions, k=len(sequence.actions)),
        ]

        remaining = count - len(actions)
        if remaining > 0:
            all_single = list(PoseType) + list(OcclusionType) + list(MovementType)
            extras = random.sample(all_single, min(remaining, len(all_single)))
            actions.extend(extras)

        random.shuffle(actions)
        return actions

    def get_current_action(self):
        if self.current_index >= len(self.actions):
            return None
        return self.actions[self.current_index]

    def mark_current_completed(self):
        action = self.get_current_action()
        if action is not None:
            self.current_index += 1
        return action

    def is_finished(self):
        return self.current_index >= len(self.actions)

    def completed_count(self):
        return min(self.current_index, len(self.actions))

    def total_actions(self):
        return len(self.actions)
