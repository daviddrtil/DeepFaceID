import cv2
import numpy as np
from mediapipe import solutions
from interactive.action_enum import PoseAction, OcclusionAction, ExpressionAction


class MetricCalculators:
    YAW_THRESHOLD_DEG = 25
    PITCH_THRESHOLD_DEG = 20
    ROLL_THRESHOLD_DEG = 20

    SINGLE_EYE_BLINK_SCORE = 0.4
    SMILE_SCORE = 0.4
    JAW_OPEN_FOR_SMILE_WITH_TEETH = 0.1
    OPEN_MOUTH_SCORE = 0.4
    EYEBROW_UP_SCORE = 0.4
    EYEBROW_DOWN_SCORE = 0.3
    GAZE_THRESHOLD = 0.4

    OCCLUSION_RATIO_THRESHOLD = 0.5     # 50 percentage of face region
    HAND_OVERLAP_THRESHOLD = 0.01
    HAND_MASK_DILATE_RATIO = 0.023
    PALM_DOWN_EXTENSION_RATIO = 0.15
    PALM_SIDE_EXTENSION_RATIO = 0.4
    HAND_BONE_THICKNESS_PX = 1
    PALM_INDICES = [0, 1, 5, 9, 13, 17, 18]

    CUSTOM_NOSE_CONNECTIONS = list(solutions.face_mesh.FACEMESH_NOSE) + [(168, 48), (168, 278)]

    def __init__(self):
        self.face_3d = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        self.key_landmarks = [1, 199, 33, 263, 61, 291]
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        self.regions = {
            OcclusionAction.COVER_LEFT_EYE: list(set([p for edge in solutions.face_mesh.FACEMESH_LEFT_EYE for p in edge])),
            OcclusionAction.COVER_RIGHT_EYE: list(set([p for edge in solutions.face_mesh.FACEMESH_RIGHT_EYE for p in edge])),
            OcclusionAction.COVER_MOUTH: list(set([p for edge in solutions.face_mesh.FACEMESH_LIPS for p in edge])),
            OcclusionAction.COVER_NOSE: list(set([p for edge in self.CUSTOM_NOSE_CONNECTIONS for p in edge])),
        }

    def evaluate(self, face_result, hand_result, mp_image, original_w, original_h):
        actions = {
            "pose": [],
            "occlusions": [],
            "expressions": [],
            "yaw": None,
            "pitch": None,
            "roll": None,
            "face_detected": False,
            "hand_detected": False,
            "hand_face_overlap": False,
        }
        small_w = mp_image.width
        small_h = mp_image.height

        hand_mask_small = self._generate_hand_mask(hand_result.hand_landmarks, small_w, small_h)

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            actions["face_detected"] = True
            actions["yaw"], actions["pitch"], actions["roll"] = self._get_face_orientation(face_landmarks, small_w, small_h)
            actions["pose"] = self._get_head_pose(actions["yaw"], actions["pitch"], actions["roll"])

            if hand_result.hand_landmarks:
                actions["hand_detected"] = True
                actions["hand_face_overlap"] = self._is_hand_face_overlap(face_landmarks, hand_mask_small, small_w, small_h)
                actions["occlusions"] = self._get_occlusions(face_landmarks, hand_mask_small, small_w, small_h)

            if face_result.face_blendshapes:
                actions["expressions"] = self._get_expressions(face_result.face_blendshapes[0])

        hand_mask_large = cv2.resize(hand_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return actions, hand_mask_large

    def _generate_hand_mask(self, hand_landmarks_list, w, h):
        mask = np.zeros((h, w), dtype=np.uint8)
        if not hand_landmarks_list:
            return mask

        for hand_landmarks in hand_landmarks_list:
            self._draw_palm_region(mask, hand_landmarks, w, h)
            self._draw_hand_connections(mask, hand_landmarks, w, h)

        return self._dilate_hand_mask(mask, w)

    def _draw_palm_region(self, mask, hand_landmarks, w, h):
        palm_points = [[int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)] for i in self.PALM_INDICES]
        extended_palm_points = self._extend_palm_points(palm_points)
        palm_array = np.array(extended_palm_points, dtype=np.int32)
        palm_hull = cv2.convexHull(palm_array)
        cv2.fillPoly(mask, [palm_hull], 255)

    def _extend_palm_points(self, palm_points):
        x0, y0 = palm_points[0]
        x9, y9 = palm_points[3]
        dx_down, dy_down = x0 - x9, y0 - y9

        x5, y5 = palm_points[2]
        x17, y17 = palm_points[5]
        dx_across, dy_across = x5 - x17, y5 - y17

        ext_x = x0 + dx_down * self.PALM_DOWN_EXTENSION_RATIO
        ext_y = y0 + dy_down * self.PALM_DOWN_EXTENSION_RATIO
        ext1_x = int(ext_x + dx_across * self.PALM_SIDE_EXTENSION_RATIO)
        ext1_y = int(ext_y + dy_across * self.PALM_SIDE_EXTENSION_RATIO)
        ext2_x = int(ext_x - dx_across * self.PALM_SIDE_EXTENSION_RATIO)
        ext2_y = int(ext_y - dy_across * self.PALM_SIDE_EXTENSION_RATIO)

        dx_pinky = x17 - x5
        dy_pinky = y17 - y5
        ext3_x = int(x17 + dx_pinky * self.PALM_DOWN_EXTENSION_RATIO)
        ext3_y = int(y17 + dy_pinky * self.PALM_DOWN_EXTENSION_RATIO)

        return palm_points + [[ext1_x, ext1_y], [ext2_x, ext2_y], [ext3_x, ext3_y]]

    def _draw_hand_connections(self, mask, hand_landmarks, w, h):
        lines = []
        for connection in solutions.hands.HAND_CONNECTIONS:
            pt1, pt2 = hand_landmarks[connection[0]], hand_landmarks[connection[1]]
            lines.append([[int(pt1.x * w), int(pt1.y * h)], [int(pt2.x * w), int(pt2.y * h)]])
        cv2.polylines(mask, np.array(lines, dtype=np.int32), False, 255, thickness=self.HAND_BONE_THICKNESS_PX)

    def _dilate_hand_mask(self, mask, w):
        kernel_size = max(3, int(w * self.HAND_MASK_DILATE_RATIO))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(mask, kernel, iterations=1)

    def _is_hand_face_overlap(self, face_landmarks, hand_mask, w, h):
        pts_x = np.array([int(lm.x * w) for lm in face_landmarks])
        pts_y = np.array([int(lm.y * h) for lm in face_landmarks])

        valid = (pts_x >= 0) & (pts_x < w) & (pts_y >= 0) & (pts_y < h)
        valid_x, valid_y = pts_x[valid], pts_y[valid]

        if len(valid_x) == 0:
            return False

        points_inside = np.sum(hand_mask[valid_y, valid_x] == 255)
        overlap =  points_inside / len(valid_x)
        return overlap > 0.01

    def _get_occlusions(self, face_landmarks, hand_mask, w, h):
        occluded = []
        for action_type, indices in self.regions.items():
            pts_x = np.array([int(face_landmarks[idx].x * w) for idx in indices])
            pts_y = np.array([int(face_landmarks[idx].y * h) for idx in indices])

            valid = (pts_x >= 0) & (pts_x < w) & (pts_y >= 0) & (pts_y < h)
            valid_x, valid_y = pts_x[valid], pts_y[valid]
            if len(valid_x) == 0:
                continue

            points_inside = np.sum(hand_mask[valid_y, valid_x] == 255)
            if points_inside / len(indices) > self.OCCLUSION_RATIO_THRESHOLD:
                occluded.append(action_type)
        return occluded

    def _get_face_orientation(self, landmarks, w, h):
        face_2d = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in self.key_landmarks], dtype=np.float64)
        cam_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)

        _, rot_vec, _ = cv2.solvePnP(self.face_3d, face_2d, cam_matrix, self.dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch, yaw, roll = angles[0], angles[1], angles[2]
        if pitch < -90: pitch += 180
        elif pitch > 90: pitch -= 180

        if roll <= -90: roll += 180
        elif roll > 90: roll -= 180

        return yaw, pitch, roll

    def _get_head_pose(self, yaw, pitch, roll):
        pose_actions = []
        if yaw >= self.YAW_THRESHOLD_DEG:
            pose_actions.append(PoseAction.MOVE_HEAD_RIGHT)
        elif yaw <= -self.YAW_THRESHOLD_DEG:
            pose_actions.append(PoseAction.MOVE_HEAD_LEFT)

        if pitch >= self.PITCH_THRESHOLD_DEG:
            pose_actions.append(PoseAction.MOVE_HEAD_DOWN)
        elif pitch <= -self.PITCH_THRESHOLD_DEG:
            pose_actions.append(PoseAction.MOVE_HEAD_UP)

        if roll >= self.ROLL_THRESHOLD_DEG:
            pose_actions.append(PoseAction.LEAN_HEAD_LEFT)
        elif roll <= -self.ROLL_THRESHOLD_DEG:
            pose_actions.append(PoseAction.LEAN_HEAD_RIGHT)

        return pose_actions

    def _get_expressions(self, blendshapes):
        scores = {cat.category_name: cat.score for cat in blendshapes}
        expr = []

        left_blink = scores.get("eyeBlinkLeft", 0) > self.SINGLE_EYE_BLINK_SCORE
        right_blink = scores.get("eyeBlinkRight", 0) > self.SINGLE_EYE_BLINK_SCORE
        if left_blink or right_blink:
            expr.append(ExpressionAction.BLINK)
        if left_blink:
            expr.append(ExpressionAction.BLINK_LEFT_EYE)
        elif right_blink:
            expr.append(ExpressionAction.BLINK_RIGHT_EYE)

        smile = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2
        jaw_open = scores.get("jawOpen", 0)
        if smile > self.SMILE_SCORE:
            if jaw_open > self.JAW_OPEN_FOR_SMILE_WITH_TEETH:
                expr.append(ExpressionAction.SMILE_TEETH)
            else:
                expr.append(ExpressionAction.SMILE)
        if jaw_open > self.OPEN_MOUTH_SCORE:
            expr.append(ExpressionAction.OPEN_MOUTH)

        brow_up = (scores.get("browInnerUp", 0) + scores.get("browOuterUpLeft", 0) + scores.get("browOuterUpRight", 0)) / 3
        brow_down = (scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)) / 2
        if brow_up > self.EYEBROW_UP_SCORE:
            expr.append(ExpressionAction.EYEBROWS_UP)
        elif brow_down > self.EYEBROW_DOWN_SCORE:
            expr.append(ExpressionAction.EYEBROWS_DOWN)

        # Unused - not user friendly
        # gaze_left = (scores.get("eyeLookOutLeft", 0) + scores.get("eyeLookInRight", 0)) / 2
        # gaze_right = (scores.get("eyeLookInLeft", 0) + scores.get("eyeLookOutRight", 0)) / 2
        # gaze_up = (scores.get("eyeLookUpLeft", 0) + scores.get("eyeLookUpRight", 0)) / 2
        # gaze_down = (scores.get("eyeLookDownLeft", 0) + scores.get("eyeLookDownRight", 0)) / 2
        # if gaze_left > self.GAZE_THRESHOLD and gaze_left > gaze_right:
        #     expr.append(MovementType.GAZE_LEFT)
        # elif gaze_right > self.GAZE_THRESHOLD and gaze_right > gaze_left:
        #     expr.append(MovementType.GAZE_RIGHT)
        # if gaze_up > self.GAZE_THRESHOLD and gaze_up > gaze_down:
        #     expr.append(MovementType.GAZE_UP)
        # elif gaze_down > self.GAZE_THRESHOLD and gaze_down > gaze_up:
        #     expr.append(MovementType.GAZE_DOWN)

        return expr
