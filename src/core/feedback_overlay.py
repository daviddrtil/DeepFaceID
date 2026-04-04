import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image, ImageDraw, ImageFont
from interactive.metric_calculators import MetricCalculators
from passive import passive_runner
import settings


class FeedbackOverlay:
    def __init__(self):
        self.font_size = None
        self.font = None
        self.face_specs = {
            "mesh": solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=0, color=(80, 80, 80)),
            "oval": solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=0, color=(200, 200, 200)),
            "eyes": solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=0, color=(255, 255, 0)),
            "lips": solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=0, color=(100, 100, 255)),
            "nose": solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=0, color=(0, 165, 255)),
        }
        self.hand_specs = {
            "joints": solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(255, 255, 255)),
            "bones": solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 165, 255)),
        }
        self.custom_nose_connections = MetricCalculators.CUSTOM_NOSE_CONNECTIONS

    @staticmethod
    def _get_font_size(frame_height):
        base_font_size = 32
        return max(16, int(base_font_size * frame_height / settings.config.base_frame_height))

    @staticmethod
    def _load_font(font_size):
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            try:
                return ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
            except OSError:
                print("Warning: Fonts not found. Using tiny default font.")
                return ImageFont.load_default()

    def _draw_face(self, frame, landmarks):
        proto = self._to_proto(landmarks)
        solutions.drawing_utils.draw_landmarks(frame, proto, solutions.face_mesh.FACEMESH_TESSELATION, None, self.face_specs["mesh"])
        solutions.drawing_utils.draw_landmarks(frame, proto, solutions.face_mesh.FACEMESH_FACE_OVAL, None, self.face_specs["oval"])
        solutions.drawing_utils.draw_landmarks(frame, proto, solutions.face_mesh.FACEMESH_LIPS, None, self.face_specs["lips"])
        solutions.drawing_utils.draw_landmarks(frame, proto, self.custom_nose_connections, None, self.face_specs["nose"])
        solutions.drawing_utils.draw_landmarks(frame, proto, solutions.face_mesh.FACEMESH_LEFT_EYE, None, self.face_specs["eyes"])
        solutions.drawing_utils.draw_landmarks(frame, proto, solutions.face_mesh.FACEMESH_RIGHT_EYE, None, self.face_specs["eyes"])

    def _draw_hands(self, frame, hand_landmarks_list, hand_mask):
        self._apply_hand_mask(frame, hand_mask)
        for landmarks in hand_landmarks_list:
            proto = self._to_proto(landmarks)
            solutions.drawing_utils.draw_landmarks(frame, proto, solutions.hands.HAND_CONNECTIONS, self.hand_specs["joints"], self.hand_specs["bones"])

    def _to_proto(self, landmarks):
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks)
        return proto

    def _apply_hand_mask(self, frame, hand_mask):
        mask = hand_mask == 255
        frame[mask] = frame[mask] * 0.7 + np.array([0, 255, 0]) * 0.3

    def _draw_text_line(self, frame, text, color_bgr, start_y, align="left"):
        if (self.font is None):
            self.font_size = self._get_font_size(frame.shape[0])
            self.font = self._load_font(self.font_size)

        left, top, right, bottom = self.font.getbbox(text)
        box_w = (right - left) + 20
        box_h = (bottom - top) + 20

        start_x = 20 if align == "left" else frame.shape[1] - box_w - 20
        end_y = start_y + box_h
        end_x = start_x + box_w

        if end_y < frame.shape[0] and end_x < frame.shape[1] and start_x >= 0 and start_y >= 0:
            roi = frame[start_y:end_y, start_x:end_x]
            dark_roi = cv2.addWeighted(roi, 0.4, np.zeros_like(roi), 0.6, 0)
            roi_rgb = cv2.cvtColor(dark_roi, cv2.COLOR_BGR2RGB)
            pil_box = Image.fromarray(roi_rgb)

            ImageDraw.Draw(pil_box).text(
                (10, 10), text, font=self.font, fill=(color_bgr[2], color_bgr[1], color_bgr[0])
            )

            frame[start_y:end_y, start_x:end_x] = cv2.cvtColor(np.array(pil_box), cv2.COLOR_RGB2BGR)

        return start_y + box_h + 10

    def _draw_interactive_data(self, frame, actions, top_text_offset):
        current_y = top_text_offset

        if not actions.get("face_detected"):
            current_y = self._draw_text_line(frame, "Face Not Detected", (0, 0, 255), current_y)

        yaw = actions.get("yaw")
        pitch = actions.get("pitch")
        roll = actions.get("roll")
        yaw_text = f"{yaw:.0f}" if yaw is not None else "N/A"
        pitch_text = f"{pitch:.0f}" if pitch is not None else "N/A"
        roll_text = f"{roll:.0f}" if roll is not None else "N/A"
        current_y = self._draw_text_line(
            frame,
            f"Yaw: {yaw_text} deg | Pitch: {pitch_text} deg | Roll: {roll_text} deg",
            (255, 255, 255),
            current_y,
        )

        if actions.get("hand_detected"):
            text = "Hand Detected"
            if actions.get("hand_face_overlap"):
                text += " + Hand-Face Overlap"
            current_y = self._draw_text_line(frame, text, (255, 255, 255), current_y)

        pose_actions = actions.get("pose", [])
        if pose_actions:
            pose_labels = [p.value.replace("Turn ", "").replace("Tilt ", "") for p in pose_actions]
            current_y = self._draw_text_line(frame, "Pose: " + ", ".join(pose_labels), (255, 255, 0), current_y)

        occlusions = actions.get("occlusions", [])
        if occlusions:
            occlusion_labels = [o.value.replace("Cover ", "") for o in occlusions]
            current_y = self._draw_text_line(frame, "Occlusion: " + ", ".join(occlusion_labels), (255, 255, 0), current_y)

        expressions = actions.get("expressions", [])
        if expressions:
            expression_labels = [e.value for e in expressions]
            current_y = self._draw_text_line(frame, "Expression: " + ", ".join(expression_labels), (255, 255, 0), current_y)

        return current_y

    def _draw_passive_data(self, frame, passive_score, top_text_offset):
        current_y = top_text_offset
        if passive_score is not None:
            text_color = (0, 0, 255) if passive_score > passive_runner.DEEPFAKE_SCORE_THRESHOLD else (0, 255, 0)
            current_y = self._draw_text_line(frame, f"Passive Score: {passive_score * 100:.1f} %", text_color, top_text_offset)
        return current_y

    def _get_action_progress(self, overlay):
        progress = overlay.get("challenge_progress")
        if progress is None or progress == 0:
            return ""
        return f" ({int(progress * 100)}%)"

    def _draw_action_overlay(self, frame, overlay, top_text_offset):
        current_y = top_text_offset
        if not overlay:
            return current_y

        current_action = overlay.get("current_action")
        if current_action:
            action_name = current_action.value if hasattr(current_action, 'value') else current_action
            progress = self._get_action_progress(overlay)
            current_y = self._draw_text_line(frame, f"Action: {action_name}" + progress, (255, 255, 255), current_y, align="right")

        decision = overlay.get("decision")
        decision_text = overlay.get("decision_text")
        if decision or decision_text:
            color = (0, 255, 0) if decision == "pass" else (0, 0, 255) if decision == "fail" else (255, 255, 255)
            text = decision_text if decision_text else decision.upper()
            current_y = self._draw_text_line(frame, f"{text}", color, current_y, align="right")

        done = overlay.get("completed_action")
        done_alpha = overlay.get("completed_alpha", 1.0)
        if done and done_alpha > 0:
            color = (int(255 * done_alpha), int(255 * done_alpha), 0)
            current_y = self._draw_text_line(frame, f"Completed: {done}", color, current_y, align="right")

        return current_y

    def draw(self, frame, face_result, hand_result, actions, hand_mask, passive_score, overlay=None):
        if settings.config.draw_face and face_result and face_result.face_landmarks:
            self._draw_face(frame, face_result.face_landmarks[0])

        if settings.config.draw_hands and hand_result and hand_result.hand_landmarks:
            self._draw_hands(frame, hand_result.hand_landmarks, hand_mask)

        TOP_OFFSET = 30
        self._draw_action_overlay(frame, overlay, TOP_OFFSET)
        if settings.config.debug_mode:
            new_top_offset = self._draw_passive_data(frame, passive_score, TOP_OFFSET)
            self._draw_interactive_data(frame, actions, new_top_offset)

        return frame
