from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    face_output_size: int = 256
    face_zoom_out: float = 0.25
    one_euro_min_cutoff: float = 0.01
    one_euro_beta: float = 0.1
    one_euro_d_cutoff: float = 1.0
    target_fps: int = 30
    max_face_jump_px: float = 180.0
    mediapipe_target_size: int = 480
    keypoint_indices: tuple[int, ...] = (468, 473, 1, 61, 291)
