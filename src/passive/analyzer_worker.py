import queue
import threading
from collections import OrderedDict
from preprocessing.live_video_queue import LiveVideoQueue


class AnalyzerWorker:
    def __init__(self, analyzer, queue_size):
        self.analyzer = analyzer
        self.input_queue = LiveVideoQueue(maxsize=queue_size)
        self.latest_score = None
        self.latest_frame = None
        self.score_buffer = OrderedDict()
        self.score_sum = 0.0
        self.score_count = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.stop_event = threading.Event()

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

            score = self.analyzer.run(frame_data)
            if score is not None:
                frame_count = frame_data.get("frame_count")
                with self.lock:
                    self.latest_score = score
                    self.latest_frame = frame_count
                    if frame_count is not None:
                        self.score_buffer[frame_count] = score
                    self.score_sum += score
                    self.score_count += 1

    def get_all(self, ref_frame=None):
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
            return score, frame or 0, avg, self.score_count

    def get_score_buffer(self):
        with self.lock:
            return dict(self.score_buffer)

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
