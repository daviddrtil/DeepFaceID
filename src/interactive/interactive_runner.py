import numpy as np
from interactive.action_detector import ActionDetector


class InteractiveRunner:
    def __init__(self):
        self.action_detector = ActionDetector()
        self.last_result = None

    def process_frame(self, preprocessed, current_action, challenge_timer):
        frame = preprocessed["frame"]
        timestamp_ms = preprocessed["timestamp_ms"]
        mp_image = preprocessed.get("mp_image")
        if mp_image is not None:
            height, width = frame.shape[:2]
            face_result, hand_result, actions, hand_mask, completed_action, challenge_progress = self.action_detector.process_frame(
                mp_image,
                timestamp_ms,
                width,
                height,
                current_action,
                challenge_timer,
            )
            self.last_result = {
                "face_result": face_result,
                "hand_result": hand_result,
                "actions": actions,
                "hand_mask": hand_mask,
                "completed_action": completed_action,
                "challenge_progress": challenge_progress,
            }
        elif self.last_result is not None:
            self.last_result["completed_action"] = None

        if self.last_result is None:
            h, w = frame.shape[:2]
            return {
                "face_result": None,
                "hand_result": None,
                "actions": {
                    "pose": [],
                    "occlusions": [],
                    "expressions": [],
                    "yaw": None,
                    "pitch": None,
                    "roll": None,
                    "face_detected": False,
                    "hand_detected": False,
                    "hand_face_overlap": False,
                },
                "hand_mask": None if h == 0 or w == 0 else np.zeros((h, w), dtype=np.uint8),
                "completed_action": None,
                "challenge_progress": 0.0,
            }

        return self.last_result

    def stop(self):
        self.action_detector.close()
