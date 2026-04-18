import queue
import random
import time
import torch
from collections import deque
from dataclasses import dataclass
from passive.passive_analyzer import PassiveAnalyzer, AnalyzerResult
from passive.spatial_analyzer.ucf_detector import get_ucf_detector
from passive.temporal_analyzer.cvit_detector import get_cvit_detector
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
    BATCH_SIZE = 8

    def __init__(self, queue_size):
        super().__init__(queue_size)
        self.detector = get_cvit_detector(DEVICE)
        self._tensor_buffer = []
        self._frame_buffer = []

    def predict(self, passive_input):
        pass

    def _run(self):
        last_received = time.time()
        while not self.stop_event.is_set():
            try:
                frame_data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                if self._tensor_buffer and (time.time() - last_received) > 0.5:
                    self._flush_batch()
                continue

            tensor = frame_data.get("cvit_face_tensor")
            if tensor is None:
                continue

            self._tensor_buffer.append(tensor)
            self._frame_buffer.append(frame_data.get("frame_count"))
            last_received = time.time()

            if len(self._tensor_buffer) >= self.BATCH_SIZE:
                self._flush_batch()

        if self._tensor_buffer:
            self._flush_batch()

    def _flush_batch(self):
        if not self._tensor_buffer:
            return
        probs = self.detector.predict_batch(self._tensor_buffer)
        with self.lock:
            for fc, score in zip(self._frame_buffer, probs):
                if fc is not None:
                    self.score_buffer[fc] = score
                self.score_sum += score
                self.score_count += 1
                self.score_max = max(self.score_max, score)
            self.latest_score = max(probs)
            self.latest_frame = self._frame_buffer[probs.index(max(probs))]
        self._tensor_buffer.clear()
        self._frame_buffer.clear()

    def reset(self):
        super().reset()
        self._tensor_buffer = []
        self._frame_buffer = []


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
