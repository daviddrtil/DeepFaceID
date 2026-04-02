import cv2
import mediapipe as mp
from preprocessing.face_aligner import FaceAligner
from preprocessing.one_euro_filter import OneEuroFilter
from preprocessing.preprocessing_config import PreprocessingConfig


class Preprocessor:
    def __init__(self):
        self.cfg = PreprocessingConfig()
        self.aligner = FaceAligner(self.cfg.face_output_size, self.cfg.keypoint_indices)
        self.stabilizer = OneEuroFilter(
            self.cfg.one_euro_min_cutoff,
            self.cfg.one_euro_beta,
            self.cfg.one_euro_d_cutoff,
        )
        self.prev_center = None
        self.inference_size = None
        self.frame_step = 1
        self.keypoint_indices = self.cfg.keypoint_indices

    @staticmethod
    def calculate_inference_size(original_width, original_height, target_width):
        inf_h = int((target_width / original_width) * original_height)
        return target_width, inf_h

    @staticmethod
    def prepare_for_mediapipe(frame, inf_w, inf_h):
        small_frame = cv2.resize(frame, (inf_w, inf_h))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return mp_image

    def configure_for_video(self, source_width, source_height, source_fps):
        self.inference_size = (int(source_width), int(source_height))
        target_fps = max(1, self.cfg.target_fps)
        source_fps = max(1.0, float(source_fps))
        self.frame_step = max(1, int(round(source_fps / target_fps)))
        self.reset_tracking()

    def reset_tracking(self):
        self.prev_center = None
        self.stabilizer.reset()

    def process_frame(self, frame, timestamp_ms, frame_count, source_fps):
        if self.inference_size is None:
            self.configure_for_video(frame.shape[1], frame.shape[0], source_fps)

        should_process = frame_count % self.frame_step == 0
        mp_image = None
        if should_process:
            target_w, target_h = self.inference_size
            mp_w, mp_h = self.calculate_inference_size(target_w, target_h, self.cfg.mediapipe_target_size)
            mp_image = self.prepare_for_mediapipe(frame, mp_w, mp_h)

        return {
            "frame": frame,
            "timestamp_ms": timestamp_ms,
            "frame_count": frame_count,
            "mp_image": mp_image,
        }

    def prepare_passive_input(self, preprocessed, face_result):
        frame = preprocessed["frame"]
        aligned_face = self.aligner.extract_and_align(frame, face_result)
        preprocessed["passive_face_input"] = None if aligned_face is None else self.aligner.preprocess_face(aligned_face)
        return preprocessed
