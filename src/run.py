import argparse
import traceback
import threading
import settings
from pathlib import Path
from utils.log_filter import LogFilter
from utils.signal_helper import install_sigint_handler
from core.liveness_detection_engine import LivenessDetectionEngine
from preprocessing.static_video_loader import StaticVideoLoader
from preprocessing.video_writer import VideoWriter
from utils.path_helper import PathHelper
from web.web_server import WebServer
from web.web_socket_input import WebSocketInput
from web.web_output import WebOutput


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Real-time liveness detection for facial deepfake verification.")
    parser.add_argument("--input-video", type=str, nargs="?", default=project_root / "recordings" / "real_daviddrtil_cover_eye.mp4", help="Path to the input video file (used only in --no-live mode).")
    parser.add_argument("--output-root-dir", type=str, nargs="?", default=project_root / "outputs", help="Base path to the output directory.")
    parser.add_argument("--stats-filename", type=str, default="stats.txt", help="Filename for the output statistics text file.")
    parser.add_argument("--live", action="store_true", default=True, help="Run the web-based live verification (default: enabled).")
    parser.add_argument("--no-live", action="store_false", dest="live", help="Disable live mode and process static video instead.")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode to display additional overlay and verbose logging (default: enabled).")
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="Disable debug overlays and verbose logging.")
    parser.add_argument("--no-saving-output", action="store_false", dest="save_output", help="Disable saving output video and frames.")
    parser.add_argument("--frame-sampling", type=int, default=30, metavar="N", help="Save every N-th processed frame (with overlay) as JPEG. 0 = disabled. (default: 30)")
    parser.add_argument("--draw", nargs="?", const="all", default=None, choices=["all", "face", "hand"], help="Draw landmarks: None (default), --draw (all), or --draw [face|hand]")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests.")

    parser.add_argument("--web-host", type=str, default="127.0.0.1", help="WebUI IP address.")
    parser.add_argument("--web-port", type=int, default=27027, help="WebUI port.")

    args = parser.parse_args()
    settings.initialize_config(args)

    stop_event = threading.Event()
    log_filter = LogFilter()
    log_filter.start()

    install_sigint_handler(stop_event)

    try:
        if settings.config.is_live:
            video_input = WebSocketInput()
            web_output = WebOutput()
            server = WebServer(stop_event, video_input, web_output)
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
        print("Keyboard interrupt: shutting down...")
    except Exception:
        traceback.print_exc()
    finally:
        stop_event.set()
        log_filter.stop()
