import threading
import queue

class EndOfStreamError(Exception):
    pass

class VideoInput:
    def __init__(self):
        self.cap = None
        self.width = 1920
        self.height = 1080
        self.fps = 30.0
        self.is_live = False
        self.total_frames = 0

        self.queue = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is not None:
            self.thread.start()

    def get_frame(self):
        try:
            data = self.queue.get(timeout=0.01)
        except queue.Empty:
            if self.stop_event.is_set():
                raise EndOfStreamError()
            raise queue.Empty

        if data is None:
            raise EndOfStreamError()

        return data

    def stop(self):
        self.stop_event.set()

        if self.queue is not None:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
