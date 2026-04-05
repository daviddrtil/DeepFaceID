import os
import re
from datetime import datetime
from pathlib import Path

class PathHelper:
    @staticmethod
    def get_output_video_name(is_live, input_video):
        video_name = "live" if is_live else f"{Path(input_video).stem}"
        return f"{video_name}_output.mp4"

    @staticmethod
    def sanitize_session_name(name):
        safe_name = re.sub(r"\s+", "_", str(name or "").strip())
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", safe_name)
        safe_name = re.sub(r"_+", "_", safe_name).strip("._")
        return (safe_name or "session").lower()

    @staticmethod
    def get_live_session_path(base_path, session_name, deepfake_label):
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sanitized_name = PathHelper.sanitize_session_name(session_name)
        
        if deepfake_label in ('real', 'fake'):
            folder_name = f"{sanitized_name}_{deepfake_label}_{timestamp}"
        else:
            folder_name = f"{sanitized_name}_{timestamp}"
        
        return Path(base_path) / folder_name

    @staticmethod
    def get_timestamped_path(base_path):
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(base_path) / f"output_{timestamp}"
