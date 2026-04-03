import random
from interactive.action_enum import ActionType


class ChallengeGenerator:
    def __init__(self, actions_count=5):
        self.actions = random.sample(list(ActionType), actions_count)
        random.shuffle(self.actions)
        self.current_index = 0

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
