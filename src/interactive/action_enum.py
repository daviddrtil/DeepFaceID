from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TypeAlias

class BaseAction(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.replace('_', ' ').title()

class PoseType(BaseAction):
    MOVE_HEAD_LEFT = auto()     # TURN_HEAD_LEFT  | LOOK_LEFT
    MOVE_HEAD_RIGHT = auto()    # TURN_HEAD_RIGHT | LOOK_RIGHT
    MOVE_HEAD_UP = auto()       # TILT_HEAD_UP    | LOOK_UP
    MOVE_HEAD_DOWN = auto()     # TILT_HEAD_DOWN  | LOOK_DOWN
    LEAN_HEAD_LEFT = auto()     # ROLL_HEAD_LEFT
    LEAN_HEAD_RIGHT = auto()    # ROLL_HEAD_RIGHT

class OcclusionType(BaseAction):
    COVER_LEFT_EYE = auto()
    COVER_RIGHT_EYE = auto()
    COVER_MOUTH = auto()
    COVER_NOSE = auto()

class MovementType(BaseAction):
    # Eye movements
    BLINK = auto()      # blink any eye or both
    BLINK_LEFT_EYE = auto()
    BLINK_RIGHT_EYE = auto()

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

ActionType: TypeAlias = PoseType | OcclusionType | MovementType

@dataclass
class ActionSet:
    # Concurrent actions happening at the exact same time
    actions: set[ActionType]
    name: str = field(init=False)

    def __post_init__(self):
        sorted_actions = sorted(self.actions, key=lambda a: a.value)
        self.name = " + ".join(a.value for a in sorted_actions)

ChallengeType: TypeAlias = ActionType | ActionSet

COMPLEX_ACTIONS = [
    ActionSet({MovementType.BLINK, MovementType.SMILE}),
    ActionSet({OcclusionType.COVER_LEFT_EYE, PoseType.MOVE_HEAD_LEFT}),
    ActionSet({OcclusionType.COVER_RIGHT_EYE, PoseType.MOVE_HEAD_RIGHT}),
    ActionSet({OcclusionType.COVER_MOUTH, PoseType.MOVE_HEAD_UP}),
    ActionSet({OcclusionType.COVER_MOUTH, PoseType.MOVE_HEAD_DOWN}),
    ActionSet({OcclusionType.COVER_MOUTH, PoseType.MOVE_HEAD_LEFT}),
    ActionSet({OcclusionType.COVER_MOUTH, PoseType.MOVE_HEAD_RIGHT}),
    ActionSet({OcclusionType.COVER_MOUTH, OcclusionType.COVER_NOSE}),
]


@dataclass
class ActionSequence:
    # Actions happening one after another
    actions: list[ActionType]
    name: str = field(init=False)

    def __post_init__(self):
        sorted_actions = sorted(self.actions, key=lambda a: a.value)
        self.name = " -> ".join(a.value for a in sorted_actions)

ACTION_SEQUENCES = [
    ActionSequence([MovementType.BLINK_LEFT_EYE, MovementType.BLINK_RIGHT_EYE]),
    ActionSequence([PoseType.MOVE_HEAD_LEFT, PoseType.MOVE_HEAD_RIGHT]),
    ActionSequence([PoseType.MOVE_HEAD_UP, PoseType.MOVE_HEAD_DOWN]),
]

def get_action_name(action):
    if action is None:
        return None
    if isinstance(action, (ActionSet, ActionSequence)):
        return action.name
    return action.value
