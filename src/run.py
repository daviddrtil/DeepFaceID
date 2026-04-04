import signal
import argparse
import traceback
import threading
import settings
from pathlib import Path
from utils.log_filter import LogFilter
from core.liveness_detection_engine import LivenessDetectionEngine
from preprocessing.static_video_loader import StaticVideoLoader
from preprocessing.video_writer import VideoWriter
from utils.path_helper import PathHelper
from web.web_server import WebServer, WebSocketInput, WebOutput


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Process video for face and hand actions.")
    # parser.add_argument("--input_video", type=str, nargs="?", default=PathHelper.get_absolute_path("Users/daviddrtil/docs/school/ing/thesis/recordings/swaps/deepfacelive/Tom_Cruise/Tom_Cruise_cover_eye.mp4"), help="Path to the input video file.")  # TODO: only for testing
    parser.add_argument("--input-video", type=str, nargs="?", default=PathHelper.get_absolute_path("Users/daviddrtil/docs/school/ing/thesis/recordings/targets/03_cover_eye.mp4"), help="Path to the input video file.")
    parser.add_argument("--output-dir", type=str, nargs="?", default=None, help="Path to the output directory. Default: outputs/YYYY-MM-DD_HH-MM-SS")
    parser.add_argument("--stats-filename", type=str, default="stats.txt", help="Filename for the output statistics text file.")
    parser.add_argument("--live", action="store_true", help="Run the web-based live verification instead of processing a static video.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to display additional overlay and verbose logging.")
    parser.add_argument("--no-saving-output", action="store_false", dest="save_output", help="Disable saving output video and frames.")
    parser.add_argument("--frame-sampling", type=int, default=30, metavar="N", help="Save every N-th processed frame (with overlay) as JPEG. 0 = disabled. (default: 30)")
    parser.add_argument("--draw", nargs="?", const="all", default=None, choices=["all", "face", "hand"], help="Draw landmarks: None (default), --draw (all), or --draw [face|hand]")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")

    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="WebUI IP address.")
    parser.add_argument("--web-port", type=int, default=27027, help="WebUI port.")

    args = parser.parse_args()

    # TODO: this can be definitelly done cleaner
    if args.output_dir is None:
        args.output_dir = str(PathHelper.get_timestamped_path(project_root / "outputs"))

    settings.initialize_config(args)

    stop_event = threading.Event()
    log_filter = LogFilter()
    log_filter.start()

    # TODO: refactor, maybe move to different place and remove the sig, frame parameters if not needed
    def sigint_handler(sig, frame):
        print("Ctrl+C pressed, stopping...")
        stop_event.set()
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        if settings.config.is_live:
            video_input = WebSocketInput()
            web_output = WebOutput()

            # TODO: this can be cleaner, maybe move engine creation to a different place
            def create_engine():
                session_output_dir = str(PathHelper.get_timestamped_path(project_root / "outputs"))
                engine_stop = threading.Event()
                return LivenessDetectionEngine(
                    video_input=video_input,
                    output_modules=[web_output],
                    stop_event=engine_stop,
                    web_output=web_output,
                    output_dir=session_output_dir,
                )

            server = WebServer(stop_event, video_input, web_output, create_engine)
            server.start()
            print("Press Ctrl+C to stop the server.")
            while not stop_event.is_set():
                try:
                    stop_event.wait(timeout=1.0)
                except KeyboardInterrupt:
                    break
        else:
            video_input = StaticVideoLoader()
            output_modules = [VideoWriter(width=video_input.width, height=video_input.height, fps=video_input.fps)]
            engine = LivenessDetectionEngine(
                video_input=video_input,
                output_modules=output_modules,
                stop_event=stop_event,
            )
            engine.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt: Shutting down...")
    except Exception:
        traceback.print_exc()
    finally:
        stop_event.set()
        log_filter.stop()
