import concurrent.futures
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_ROOT = Path(__file__).parent
_FACE_MODEL = _ROOT / "mediapipe_tasks" / "face_landmarker.task"
_HAND_MODEL = _ROOT / "mediapipe_tasks" / "hand_landmarker.task"


class LandmarkExtractor:
    _face_model_buffer = None
    _hand_model_buffer = None

    def __init__(self, face_model_path=_FACE_MODEL, hand_model_path=_HAND_MODEL):
        if LandmarkExtractor._face_model_buffer is None:
            LandmarkExtractor._face_model_buffer = Path(face_model_path).read_bytes()
        if LandmarkExtractor._hand_model_buffer is None:
            LandmarkExtractor._hand_model_buffer = Path(hand_model_path).read_bytes()
        self._create_landmarkers()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _create_landmarkers(self):
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=LandmarkExtractor._face_model_buffer),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=LandmarkExtractor._hand_model_buffer),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    def detect(self, mp_image, timestamp_ms):
        future_face = self.executor.submit(self.face_landmarker.detect_for_video, mp_image, timestamp_ms)
        future_hand = self.executor.submit(self.hand_landmarker.detect_for_video, mp_image, timestamp_ms)
        return future_face.result(), future_hand.result()

    def reset(self):
        self.face_landmarker.close()
        self.hand_landmarker.close()
        self._create_landmarkers()

    def close(self):
        self.executor.shutdown(wait=True)
        self.face_landmarker.close()
        self.hand_landmarker.close()
