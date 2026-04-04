import threading
import time
from queue import Empty
from preprocessing.video_input import VideoInput, EndOfStreamError
from preprocessing.live_video_queue import LiveVideoQueue


class WebSocketInput(VideoInput):
    def __init__(self):
        super().__init__()
        self.width = None
        self.height = None
        self.fps = 30.0
        self.is_live = True
        self.queue = LiveVideoQueue(maxsize=4)
        self.frame_count = 0
        self.start_time = None
        self.connected = threading.Event()

    def print_video_info(self):
        pass

    def put_frame(self, frame, width, height):
        if self.start_time is None:
            self.start_time = time.time()
            self.width = width
            self.height = height
            print(f"WebSocket Input: {width}x{height} at {self.fps:.0f} fps")
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        self.queue.put_latest((frame, timestamp_ms, self.frame_count))
        self.frame_count += 1

    def get_frame(self):
        try:
            data = self.queue.get(timeout=0.5)
        except Empty:
            if self.stop_event.is_set():
                raise EndOfStreamError()
            raise Empty
        if data is None:
            raise EndOfStreamError()
        return data

    def reset(self):
        self.frame_count = 0
        self.start_time = None
        self.width = None
        self.height = None
        self.stop_event.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
