from enum import Enum, auto

class ActionType(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.replace('_', ' ').title()

    # Poses
    TURN_HEAD_LEFT = auto()
    TURN_HEAD_RIGHT = auto()
    TILT_HEAD_UP = auto()
    TILT_HEAD_DOWN = auto()

    # Occlusions
    COVER_LEFT_EYE = auto()
    COVER_RIGHT_EYE = auto()
    COVER_MOUTH = auto()
    COVER_NOSE = auto()

    # Expressions
    BLINK = auto()
    SMILE = auto()
    SMILE_WITH_TEETH = auto()
    OPEN_MOUTH = auto()
    #TONGUE_OUT = auto()    # TODO: fix detection
