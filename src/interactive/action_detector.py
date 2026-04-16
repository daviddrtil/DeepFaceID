from interactive.action_validator import ActionValidator
from interactive.landmark_extractor import LandmarkExtractor
from interactive.metric_calculators import MetricCalculators


class ActionDetector:
    def __init__(self):
        self.landmark_extractor = LandmarkExtractor()
        self.metric_calculators = MetricCalculators()
        self.action_validator = ActionValidator()

    def process_frame(self, mp_image, timestamp_ms, original_w, original_h, current_action, challenge_timer):
        face_result, hand_result = self.landmark_extractor.detect(mp_image, timestamp_ms)
        actions, hand_mask_large = self.metric_calculators.evaluate(face_result, hand_result, mp_image, original_w, original_h)
        completed_action, challenge_progress = self.action_validator.validate(current_action, actions, challenge_timer, timestamp_ms)
        return face_result, hand_result, actions, hand_mask_large, completed_action, challenge_progress

    def reset(self):
        self.landmark_extractor.reset()
        self.action_validator = ActionValidator()

    def close(self):
        self.landmark_extractor.close()
