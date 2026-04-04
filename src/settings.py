from dataclasses import dataclass
from pathlib import Path
from utils.path_helper import PathHelper

# Global instance
config = None

@dataclass(frozen=True)
class PipelineConfig:
    input_video_path: str
    output_dir: str
    output_video_path: str
    output_stats_path: str
    is_live: bool
    save_output: bool
    frame_sampling: int
    draw_face: bool
    draw_hands: bool
    debug_mode: bool
    max_frames: int | None

    web_host: str
    web_port: int

    base_frame_width: int = 1920
    base_frame_height: int = 1080
    base_video_fps: int = 30
    input_queue_size: int = 30

def initialize_config(args):
    global config

    config = PipelineConfig(
        input_video_path=args.input_video,
        output_dir=args.output_dir,
        output_video_path = PathHelper.get_output_video_path(args),
        output_stats_path = Path(args.output_dir) / args.stats_filename,
        is_live=args.live,
        save_output=args.save_output,
        frame_sampling=args.frame_sampling,
        draw_face=args.draw in ["all", "face"],
        draw_hands=args.draw in ["all", "hand"],
        debug_mode=args.debug,
        max_frames=args.max_frames,
        web_host=args.web_host,
        web_port=args.web_port,
    )
