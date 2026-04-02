import concurrent.futures
import os
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FACE_MODEL = os.path.join(ROOT_DIR, "mediapipe_tasks/face_landmarker.task")
DEFAULT_HAND_MODEL = os.path.join(ROOT_DIR, "mediapipe_tasks/hand_landmarker.task")


class LandmarkExtractor:
    def __init__(self, face_model_path=DEFAULT_FACE_MODEL, hand_model_path=DEFAULT_HAND_MODEL):
        face_model_buffer = self._read_model_bytes(face_model_path)
        base_options_face = python.BaseOptions(model_asset_buffer=face_model_buffer)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_face)

        hand_model_buffer = self._read_model_bytes(hand_model_path)
        base_options_hand = python.BaseOptions(model_asset_buffer=hand_model_buffer)
        options_hand = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _read_model_bytes(self, model_path):
        path_obj = Path(model_path)
        if not path_obj.is_absolute():
            path_obj = (Path(ROOT_DIR) / path_obj).resolve()
        return path_obj.read_bytes()

    def detect(self, mp_image, timestamp_ms):
        future_face = self.executor.submit(self.face_landmarker.detect_for_video, mp_image, timestamp_ms)
        future_hand = self.executor.submit(self.hand_landmarker.detect_for_video, mp_image, timestamp_ms)
        return future_face.result(), future_hand.result()

    def close(self):
        self.executor.shutdown(wait=True)
        self.face_landmarker.close()
        self.hand_landmarker.close()
