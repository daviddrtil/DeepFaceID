import cv2
import threading
import time
import queue
import settings
from preprocessing.live_video_queue import LiveVideoQueue
from preprocessing.video_input import VideoInput

class UdpLiveStreamLoader(VideoInput):
    def __init__(self):
        super().__init__()
        print("Initializing udp live stream...")
        self.udp_url = f"udp://{settings.config.stream_host}:{settings.config.stream_port}?fflags=nobuffer&flags=low_delay&pkt_size=1316"
        self.cap = cv2.VideoCapture(self.udp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = self.cap.read()
        if ret:
            self.height, self.width = frame.shape[:2]

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
        self.is_live = True

        self.queue = LiveVideoQueue(maxsize=4)
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)

    def print_video_info(self):
        video_info = (
            f"Live Stream Info:\n"
            f"URL              : {self.udp_url}\n"
            f"Resolution       : {self.width}x{self.height} ({self.fps:.0f} fps)\n"
        )
        print(video_info)

    def _reader_thread(self):
        frame_count = 0
        start_time = time.time()
        cap = self.cap 
        while not self.stop_event.is_set() and cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            timestamp_ms = int((time.time() - start_time) * 1000)
            self.queue.put_latest((frame, timestamp_ms, frame_count))
            frame_count += 1

    def get_frame(self):
        data = super().get_frame()
        while not self.queue.empty():
            try:
                data = self.queue.get_nowait()
            except queue.Empty:
                break
        return data
