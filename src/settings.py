from dataclasses import dataclass, replace
from pathlib import Path

from utils.path_helper import PathHelper

# Global instance
config = None


@dataclass(frozen=True)
class PipelineConfig:
    input_video_path: str
    output_root_dir: str
    output_dir: str
    stats_filename: str
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
    deepfake_label: str | None = None

def initialize_config(args):
    global config

    if args.live:
        output_dir = Path(args.output_root_dir)
    else:
        output_dir = PathHelper.get_timestamped_path(args.output_root_dir)

    output_video_path = output_dir / PathHelper.get_output_video_name(args.live, args.input_video)
    output_stats_path = output_dir / args.stats_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        input_video_path=args.input_video,
        output_root_dir=str(output_dir),
        output_dir=str(output_dir),
        stats_filename=args.stats_filename,
        output_video_path=str(output_video_path),
        output_stats_path=str(output_stats_path),
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

def set_output_dir(output_dir, deepfake_label):
    global config

    output_dir_path = Path(output_dir)
    output_video_path = output_dir_path / PathHelper.get_output_video_name(config.is_live, config.input_video_path)
    output_stats_path = output_dir_path / config.stats_filename

    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = replace(
        config,
        output_dir=str(output_dir_path),
        output_video_path=str(output_video_path),
        output_stats_path=str(output_stats_path),
        deepfake_label=deepfake_label
    )

def set_live_session_output(session_name, deepfake_label):
    session_output_dir = PathHelper.get_live_session_path(config.output_root_dir, session_name, deepfake_label)
    set_output_dir(session_output_dir, deepfake_label)
    return session_output_dir
