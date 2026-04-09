import queue
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from preprocessing.live_video_queue import LiveVideoQueue


@dataclass
class AnalyzerResult:
    current_score: float = None
    current_frame: int = 0
    avg_score: float = None
    total_count: int = 0


class PassiveAnalyzer(ABC):
    def __init__(self, queue_size):
        self.input_queue = LiveVideoQueue(maxsize=queue_size)
        self.latest_score = None
        self.latest_frame = None
        self.score_buffer = OrderedDict()
        self.score_sum = 0.0
        self.score_count = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.stop_event = threading.Event()

    @abstractmethod
    def predict(self, passive_input):
        ...

    def start(self):
        self.thread.start()

    def submit(self, frame_data):
        self.input_queue.put_latest(frame_data)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame_data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            score = self.predict(frame_data)
            if score is not None:
                frame_count = frame_data.get("frame_count")
                with self.lock:
                    self.latest_score = score
                    self.latest_frame = frame_count
                    if frame_count is not None:
                        self.score_buffer[frame_count] = score
                    self.score_sum += score
                    self.score_count += 1

    def get_result(self, ref_frame=None):
        with self.lock:
            if ref_frame is not None:
                selected_frame, selected_score = None, None
                for frame, score in self.score_buffer.items():
                    if frame <= ref_frame and (selected_frame is None or frame > selected_frame):
                        selected_frame = frame
                        selected_score = score
                score, frame = selected_score, selected_frame
            else:
                score, frame = self.latest_score, self.latest_frame
            avg = (self.score_sum / self.score_count) if self.score_count else None
            return AnalyzerResult(score, frame or 0, avg, self.score_count)

    def get_score_buffer(self):
        with self.lock:
            return dict(self.score_buffer)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
