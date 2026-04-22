from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import torch
from skimage import transform
import settings


class FaceAligner:
    def __init__(self, output_size, keypoint_indices):
        self.output_size = output_size
        self.keypoint_indices = keypoint_indices
        self.dst_pts = self._get_aligned_reference_points(self.output_size)

        self._debug_face_index = 0
        base_output_dir = Path(settings.config.output_dir)
        self._debug_output_dir = base_output_dir / "preprocessed_faces"
        # if settings.config.debug_mode:
        #     self._debug_output_dir.mkdir(parents=True, exist_ok=True)

    def _save_debug_aligned_face(self, aligned_face):
        if not settings.config.debug_mode or aligned_face is None:
            return
        if self._debug_face_index % 10 != 0:
            self._debug_face_index += 1
            return
        output_path = self._debug_output_dir / f"face{self._debug_face_index:04d}.jpg"
        aligned_face.save(output_path)
        self._debug_face_index += 1

    @staticmethod
    def _get_aligned_reference_points(output_size):
        outsize = [output_size, output_size]
        scale = 1.3
        target_size = [112, 112]

        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        dst[:, 0] += 8.0
        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize
        margin_rate = scale - 1.0
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)
        return dst

    @staticmethod
    def _normalize_points(src_pts):
        left_eye, right_eye, nose, left_mouth, right_mouth = src_pts.tolist()
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        if left_mouth[0] > right_mouth[0]:
            left_mouth, right_mouth = right_mouth, left_mouth
        return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)

    def _align(self, frame_rgb, src_pts):
        src_pts = self._normalize_points(np.asarray(src_pts, dtype=np.float32))
        try:
            tform = transform.SimilarityTransform.from_estimate(src_pts, self.dst_pts)
        except ValueError:
            return None
        matrix = tform.params[0:2, :]
        aligned_face = cv2.warpAffine(frame_rgb, matrix, (self.output_size, self.output_size), flags=cv2.INTER_CUBIC)
        return Image.fromarray(aligned_face)

    def extract_and_align(self, frame_bgr, face_result):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape

        if face_result is None or not face_result.face_landmarks:
            return None

        landmarks = face_result.face_landmarks[0]
        src_pts = np.array(
            [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in self.keypoint_indices],
            dtype=np.float32,
        )
        aligned_face = self._align(frame_rgb, src_pts)
        # if settings.config.debug_mode:
        #     self._save_debug_aligned_face(aligned_face)
        return aligned_face

    def preprocess_face(self, face_image):
        face_np = np.array(face_image, dtype=np.float32)
        face_np = face_np / 255.0
        face_np = (face_np - 0.5) / 0.5
        face_np = np.transpose(face_np, (2, 0, 1))
        return torch.from_numpy(face_np).unsqueeze(0)
