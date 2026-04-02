import os
import re
import platform
from datetime import datetime
from pathlib import Path


class PathHelper:
    @staticmethod
    def get_timestamped_path(base_path):
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return Path(base_path) / f"output_{timestamp}"

    @staticmethod
    def get_versioned_path(base_path, subdir_prefix):
        os.makedirs(base_path, exist_ok=True)

        # List all directories that match the pattern 'output' followed by numbers
        existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(Path(base_path) / d)]
        numbers = [int(re.search(rf'{subdir_prefix}(\+?\d+)', d).group(1)) for d in existing_dirs if re.search(rf'{subdir_prefix}(\d+)', d)]

        next_version = max(numbers) + 1 if numbers else 1
        return Path(base_path) / f"{subdir_prefix}{next_version:02d}"

    @staticmethod
    def get_absolute_path(relative_path):
        base = Path("C:/") if platform.system() == "Windows" else Path("/mnt/c")
        return base / relative_path

    @staticmethod
    def get_output_video_path(args):
        video_name = "live" if args.live else f"{Path(args.input_video).stem}"
        return Path(args.output_dir) / f"{video_name}_output.mp4"
