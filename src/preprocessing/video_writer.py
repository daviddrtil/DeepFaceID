import cv2
import os
import settings
import subprocess
import threading
from queue import Queue
import numpy as np


class VideoWriter:
    def __init__(self, width, height, fps, output_dir=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir if output_dir else settings.config.output_dir
        self.output_video_path = os.path.join(self.output_dir, "output.mp4")

        os.makedirs(self.output_dir, exist_ok=True)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'bgr24',
            '-r', str(self.fps), '-i', '-', '-c:v',
            'libx265', '-x265-params', 'log-level=error',
            '-crf', '24', '-preset', 'fast', '-pix_fmt', 'yuv420p',
            self.output_video_path
        ]
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        self.queue = Queue(maxsize=300)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_thread)
        self.last_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def start(self):
        self.thread.start()

    def _writer_thread(self):
        try:
            while True:
                data = self.queue.get()
                if data is None:
                    break

                frame, frame_count, _ = data
                self.last_frame = frame.copy()

                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.stdin.write(frame.tobytes())
                    self.ffmpeg_process.stdin.flush()

                if frame_count >= 0 and settings.config.frame_sampling > 0 and frame_count % settings.config.frame_sampling == 0:
                    output_file = os.path.join(self.output_dir, f"frame{frame_count:05d}.jpg")
                    cv2.imwrite(output_file, frame)

        except Exception as e:
            print(f"Writer error: {e}")
        finally:
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()

    def put_frame(self, frame, frame_count, action_message=None):
        self.queue.put((frame, frame_count, action_message))

    def put_repeated_frames(self, count):
        if count <= 0:
            return
        for _ in range(count):
            self.queue.put((self.last_frame.copy(), -1, None))

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.ffmpeg_process:
            self.ffmpeg_process.wait()
        print(f"Output saved to {self.output_video_path}")
