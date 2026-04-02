import queue
import threading
from preprocessing.live_video_queue import LiveVideoQueue

class AnalyzerWorker:
    def __init__(self, analyzer, queue_size):
        self.analyzer = analyzer
        self.input_queue = LiveVideoQueue(maxsize=queue_size)
        self.latest_result = None       # TODO: test removal of initial lag: {"score": 0.5, "frame_count": 0}
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

            result = self.analyzer.run(frame_data)
            if result is not None:
                with self.lock:
                    self.latest_result = {
                        "score": result,
                        "frame_count": frame_data.get("frame_count")
                    }

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
