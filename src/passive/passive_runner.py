import random
import time
import torch
from collections import deque
from dataclasses import dataclass
from passive.passive_analyzer import PassiveAnalyzer, AnalyzerResult
from passive.spatial_analyzer.ucf_detector import get_ucf_detector
from passive.temporal_analyzer.cvit_detector import get_cvit_detector, WINDOW_SIZE, INFERENCE_STEP, FAKE_THRESHOLD
from core.decision_logic import PASSIVE_WEIGHTS


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class PassiveResult:
    spatial: AnalyzerResult
    frequency: AnalyzerResult
    temporal: AnalyzerResult
    score_cur: float = None
    score_avg: float = None
    score_max: float = None
    score_smooth: float = None

    def __post_init__(self):
        self.score_cur = self._weighted('current_score', PASSIVE_WEIGHTS)
        self.score_avg = self._weighted('avg_score', PASSIVE_WEIGHTS)
        self.score_max = self._weighted('max_score', PASSIVE_WEIGHTS)

    def _weighted(self, attr, weights):
        total, weight = 0.0, 0.0
        for key, w in weights.items():
            score = getattr(getattr(self, key), attr)
            if score is not None:
                total += score * w
                weight += w
        return total / weight if weight else None


class SpatialAnalyzer(PassiveAnalyzer):
    def __init__(self, queue_size):
        super().__init__(queue_size)
        self.detector = get_ucf_detector(DEVICE)

    def predict(self, passive_input):
        tensor = passive_input.get("passive_face_input")
        if tensor is None:
            return None
        return self.detector.predict(tensor.to(DEVICE))


class FrequencyAnalyzer(PassiveAnalyzer):
    def __init__(self, queue_size):
        super().__init__(queue_size)

    def predict(self, passive_input):
        fps = random.uniform(100, 200)
        time.sleep(1.0 / fps)
        return random.uniform(0.4, 0.6)


class TemporalAnalyzer(PassiveAnalyzer):
    def __init__(self, queue_size):
        super().__init__(queue_size)
        self.detector = get_cvit_detector(DEVICE)
        self._frame_buffer: list = []  # accumulates tensors for sliding window
        self._frames_since_last = 0    # frames received since last prediction

    def predict(self, passive_input):
        tensor = passive_input.get("cvit_face_tensor")
        if tensor is None:
            return None

        self._frame_buffer.append(tensor)
        self._frames_since_last += 1

        if len(self._frame_buffer) < WINDOW_SIZE:
            return None

        if self._frames_since_last < INFERENCE_STEP:
            return None

        # Sliding window: use last WINDOW_SIZE frames, predict every INFERENCE_STEP
        self._frames_since_last = 0
        score = self.detector.predict_window(self._frame_buffer[-WINDOW_SIZE:])
        self._frame_buffer = self._frame_buffer[-WINDOW_SIZE:]  # trim for memory
        return score

    def reset(self):
        super().reset()
        self._frame_buffer = []
        self._frames_since_last = 0


class PassiveRunner:
    def __init__(self):
        self.spatial = SpatialAnalyzer(queue_size=4)
        self.frequency = FrequencyAnalyzer(queue_size=4)
        self.temporal = TemporalAnalyzer(queue_size=8)
        self._workers = (self.spatial, self.frequency, self.temporal)
        self._score_window = deque(maxlen=10)    # smooth window

    def start(self):
        for worker in self._workers:
            worker.start()

    def submit(self, passive_input):
        if passive_input.get("passive_face_input") is not None:
            self.spatial.submit(passive_input)
            self.frequency.submit(passive_input)
        if passive_input.get("cvit_face_tensor") is not None:
            self.temporal.submit(passive_input)

    def get_passive_result(self):
        spatial = self.spatial.get_result()
        frequency = self.frequency.get_result(ref_frame=spatial.current_frame)
        temporal = self.temporal.get_result(ref_frame=spatial.current_frame)
        result = PassiveResult(spatial=spatial, frequency=frequency, temporal=temporal)
        if result.score_cur is not None:
            self._score_window.append(result.score_cur)
        if self._score_window:
            result.score_smooth = sum(self._score_window) / len(self._score_window)
        return result

    def stop(self):
        for worker in self._workers:
            worker.stop()

    def reset(self):
        for worker in self._workers:
            worker.reset()
        self._score_window.clear()

    def get_temporal_window_stats(self):
        buf = self.temporal.get_score_buffer()
        if not buf:
            return None
        windows = [v for _, v in sorted(buf.items())]
        return max(windows), sum(1 for m in windows if m >= FAKE_THRESHOLD), len(windows)
