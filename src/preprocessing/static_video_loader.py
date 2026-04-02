import cv2
import threading
import queue
import settings
from preprocessing.video_input import VideoInput


class StaticVideoLoader(VideoInput):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(settings.config.input_video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_live = False

        self.queue = queue.Queue(maxsize=60)
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)

    def print_video_info(self):
        video_info = (
            f"Video info:\n"
            f"Video            : {settings.config.input_video_path}\n"
            f"Resolution       : {self.width}x{self.height} ({self.fps:.0f} fps)\n"
            f"Total Frames     : {self.total_frames if self.total_frames > 0 else 'unknown'} "
            f"(time={self.total_frames / self.fps:.1f}s)"
        )
        print(video_info)

    def _reader_thread(self):
        frame_count = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(None)
                break
            
            timestamp_ms = int((frame_count * 1000) / self.fps)
            self.queue.put((frame, timestamp_ms, frame_count))
            frame_count += 1
