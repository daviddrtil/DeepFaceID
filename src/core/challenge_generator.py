import random
from interactive.action_enum import (
    PoseAction, OcclusionAction, ExpressionAction, ChallengeType,
    ComplexAction, HoldStillAction, COMPLEX_ACTIONS, ACTION_SEQUENCES, get_action_name
)

class ChallengeGenerator:
    def __init__(self, actions_count=6):
        list1 = [HoldStillAction(), OcclusionAction.COVER_NOSE, ExpressionAction.OPEN_MOUTH, PoseAction.MOVE_HEAD_RIGHT, PoseAction.MOVE_HEAD_LEFT, PoseAction.LEAN_HEAD_RIGHT, ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_RIGHT})]
        list2 = [HoldStillAction(), PoseAction.LEAN_HEAD_LEFT, ExpressionAction.OPEN_MOUTH, PoseAction.MOVE_HEAD_RIGHT, PoseAction.MOVE_HEAD_LEFT, ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_DOWN}), OcclusionAction.COVER_RIGHT_EYE]
        list3 = [HoldStillAction(), OcclusionAction.COVER_MOUTH, ExpressionAction.SMILE, PoseAction.MOVE_HEAD_DOWN, PoseAction.MOVE_HEAD_UP, PoseAction.MOVE_HEAD_DOWN, ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_RIGHT_EYE})]
        list4 = [HoldStillAction(), ExpressionAction.SMILE, PoseAction.MOVE_HEAD_RIGHT, PoseAction.MOVE_HEAD_LEFT, PoseAction.MOVE_HEAD_UP, PoseAction.MOVE_HEAD_DOWN, OcclusionAction.COVER_NOSE, OcclusionAction.COVER_LEFT_EYE, OcclusionAction.COVER_MOUTH, ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_RIGHT}), ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_RIGHT_EYE}), ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_RIGHT_EYE, PoseAction.MOVE_HEAD_LEFT})]
        # self.actions: list[ChallengeType] = list4
        self.actions: list[ChallengeType] = self._generate_actions(actions_count)
        self.current_index = 0
        # for _ in range(10):
        #     actions = self._generate_actions(actions_count)
        #     print(" -> ".join(get_action_name(a) for a in actions))

    def _generate_actions(self, count):
        # One of each type guaranteed
        actions = [
            random.choice(list(PoseAction)),
            random.choice(list(OcclusionAction)),
            random.choice(list(ExpressionAction)),
            random.choice(COMPLEX_ACTIONS),
        ]

        sequence = random.choice(ACTION_SEQUENCES)
        seq_actions = random.sample(sequence.actions, len(sequence.actions))

        # Fill remaining with random single actions
        all_single = list(PoseAction) + list(OcclusionAction) + list(ExpressionAction)
        for _ in range(max(0, count - len(actions) - len(seq_actions) - 1)):
            actions.append(random.choice(all_single))

        for _ in range(100):
            random.shuffle(actions)
            result = actions.copy()
            idx = random.randint(0, len(result))
            result[idx:idx] = seq_actions
            if self._no_neighbor_conflicts(result):
                return [HoldStillAction()] + result

        return [HoldStillAction()] + result

    @staticmethod
    def _no_neighbor_conflicts(actions):
        for i in range(len(actions) - 1):
            a, b = actions[i], actions[i + 1]
            if a == b:
                return False
            if isinstance(a, ComplexAction) and not isinstance(b, ComplexAction):
                if b in a.actions:
                    return False
            if isinstance(b, ComplexAction) and not isinstance(a, ComplexAction):
                if a in b.actions:
                    return False
        return True

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
