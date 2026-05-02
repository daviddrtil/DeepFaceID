from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TypeAlias

class BaseAction(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.replace('_', ' ').title()

class PoseAction(BaseAction):
    MOVE_HEAD_LEFT = auto()     # TURN_HEAD_LEFT  | LOOK_LEFT
    MOVE_HEAD_RIGHT = auto()    # TURN_HEAD_RIGHT | LOOK_RIGHT
    MOVE_HEAD_UP = auto()       # TILT_HEAD_UP    | LOOK_UP
    MOVE_HEAD_DOWN = auto()     # TILT_HEAD_DOWN  | LOOK_DOWN
    LEAN_HEAD_LEFT = auto()     # ROLL_HEAD_LEFT
    LEAN_HEAD_RIGHT = auto()    # ROLL_HEAD_RIGHT

class OcclusionAction(BaseAction):
    COVER_LEFT_EYE = auto()
    COVER_RIGHT_EYE = auto()
    COVER_MOUTH = auto()
    COVER_NOSE = auto()

class ExpressionAction(BaseAction):
    # Eye movements - not used, deepfakes cannot realibly fake this action
    # BLINK = auto()      # blink any eye or both
    # BLINK_LEFT_EYE = auto()
    # BLINK_RIGHT_EYE = auto()

    # Gaze directions - unused - not user friendly
    # GAZE_LEFT = auto()
    # GAZE_RIGHT = auto()
    # GAZE_UP = auto()
    # GAZE_DOWN = auto()

    # Mouth movements
    SMILE = auto()
    SMILE_TEETH = auto()
    OPEN_MOUTH = auto()
    #TONGUE_OUT = auto()    # not implemented

    # Eyebrow movements
    EYEBROWS_UP = auto()
    EYEBROWS_DOWN = auto()

ActionType: TypeAlias = PoseAction | OcclusionAction | ExpressionAction

@dataclass
class ComplexAction:
    # Concurrent actions happening at the exact same time
    actions: set[ActionType]
    name: str = field(init=False)

    def __post_init__(self):
        sorted_actions = sorted(self.actions, key=lambda a: a.value)
        self.name = " + ".join(a.value for a in sorted_actions)

@dataclass(frozen=True)
class HoldStillAction:
    # Used for detector calibration
    duration_seconds: float = 2.0
    name: str = field(init=False, default="HOLD_STILL")
    value: str = field(init=False, default="Hold Still")

ChallengeType: TypeAlias = ActionType | ComplexAction | HoldStillAction

COMPLEX_ACTIONS = [
    # ComplexAction({ExpressionAction.BLINK, ExpressionAction.SMILE}),
    ComplexAction({OcclusionAction.COVER_LEFT_EYE, PoseAction.MOVE_HEAD_LEFT}),
    ComplexAction({OcclusionAction.COVER_RIGHT_EYE, PoseAction.MOVE_HEAD_RIGHT}),
    ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_UP}),
    ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_DOWN}),
    ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_LEFT}),
    ComplexAction({OcclusionAction.COVER_MOUTH, PoseAction.MOVE_HEAD_RIGHT}),
    ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_NOSE}),
    ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_LEFT_EYE}),
    ComplexAction({OcclusionAction.COVER_MOUTH, OcclusionAction.COVER_RIGHT_EYE}),
]


@dataclass
class SequenceAction:
    # Actions happening one after another
    actions: list[ActionType]
    name: str = field(init=False)

    def __post_init__(self):
        sorted_actions = sorted(self.actions, key=lambda a: a.value)
        self.name = " -> ".join(a.value for a in sorted_actions)

ACTION_SEQUENCES = [
    # SequenceAction([ExpressionAction.BLINK_LEFT_EYE, ExpressionAction.BLINK_RIGHT_EYE]),
    SequenceAction([PoseAction.MOVE_HEAD_LEFT, PoseAction.MOVE_HEAD_RIGHT]),
    SequenceAction([PoseAction.MOVE_HEAD_UP, PoseAction.MOVE_HEAD_DOWN]),
]

def get_action_name(action):
    if action is None:
        return None
    if isinstance(action, (ComplexAction, SequenceAction)):
        return action.name
    return action.value


def get_action_category(action):
    if action is None:
        return None
    if isinstance(action, HoldStillAction):
        return 'calibration'
    name = type(action).__name__
    if name.endswith('Action'):
        return name.removesuffix('Action').lower()
    return None
