import os
import sys
import threading
import warnings


class LogFilter:
    _NOISY_PATTERNS = (
        "all log messages before absl::initializelog()",
        "face_landmarker_graph.cc:174",
        "created tensorflow lite xnnpack delegate for cpu",
        "inference_feedback_manager.cc:114",
        "landmark_projection_calculator.cc:186",
        "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead."
    )

    @staticmethod
    def configure_native_logging():
        # Must run before importing modules that initialize MediaPipe/TFLite runtimes.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["GLOG_minloglevel"] = "2"
        os.environ["GLOG_logtostderr"] = "1"
        os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
        warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.GetPrototype().*")

    def __init__(self):
        self._started = False
        self._original_stderr = None
        self._original_dunder_stderr = None
        self._redirected_stderr = None
        self._stderr_fd = None
        self._saved_stderr_fd = None
        self._pipe_r = None
        self._reader_thread = None
        self._leftover = b""

    def start(self):
        if self._started:
            return
        self.configure_native_logging()
        self._original_stderr = sys.stderr
        self._original_dunder_stderr = sys.__stderr__

        try:
            self._start_fd_filter()
        except OSError:
            self._restore_streams()
            self._cleanup_fd_filter_state()
            return

        self._started = True

    def stop(self):
        if not self._started:
            return

        self._restore_streams()
        self._stop_fd_filter()
        self._started = False
        self._original_stderr = None
        self._original_dunder_stderr = None

    def _start_fd_filter(self):
        stderr = self._original_stderr
        if stderr is None or not hasattr(stderr, "fileno"):
            raise OSError("stderr stream has no fileno")

        self._stderr_fd = stderr.fileno()
        self._saved_stderr_fd = os.dup(self._stderr_fd)
        self._pipe_r, pipe_w = os.pipe()
        os.dup2(pipe_w, self._stderr_fd)
        os.close(pipe_w)

        encoding = getattr(stderr, "encoding", None) or "utf-8"
        errors = getattr(stderr, "errors", None) or "backslashreplace"
        self._redirected_stderr = os.fdopen(
            os.dup(self._stderr_fd),
            "w",
            buffering=1,
            encoding=encoding,
            errors=errors,
        )
        sys.stderr = self._redirected_stderr
        sys.__stderr__ = self._redirected_stderr
        self._reader_thread = threading.Thread(target=self._forward_native_stderr, daemon=True)
        self._reader_thread.start()

    def _stop_fd_filter(self):
        if self._saved_stderr_fd is None or self._stderr_fd is None:
            return

        try:
            sys.stderr.flush()
        except Exception:
            pass

        try:
            os.dup2(self._saved_stderr_fd, self._stderr_fd)
        except OSError:
            pass

        if self._redirected_stderr is not None:
            try:
                self._redirected_stderr.close()
            except OSError:
                pass
            self._redirected_stderr = None

        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        self._cleanup_fd_filter_state()

    def _cleanup_fd_filter_state(self):
        if self._saved_stderr_fd is not None:
            try:
                os.close(self._saved_stderr_fd)
            except OSError:
                pass

        self._stderr_fd = None
        self._saved_stderr_fd = None
        self._pipe_r = None
        self._reader_thread = None
        self._leftover = b""

    def _restore_streams(self):
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        if self._original_dunder_stderr is not None:
            sys.__stderr__ = self._original_dunder_stderr

    def _forward_native_stderr(self):
        if self._pipe_r is None:
            return

        ignore_patterns = tuple(pattern.lower() for pattern in self._NOISY_PATTERNS)
        with os.fdopen(self._pipe_r, "rb", buffering=0) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                self._leftover += chunk
                while b"\n" in self._leftover:
                    line, self._leftover = self._leftover.split(b"\n", 1)
                    self._forward_native_line(line + b"\n", ignore_patterns)

            if self._leftover:
                self._forward_native_line(self._leftover, ignore_patterns)

    def _forward_native_line(self, line_bytes, ignore_patterns):
        if self._saved_stderr_fd is None:
            return

        lower_line = line_bytes.decode("utf-8", errors="replace").lower()
        if any(pattern in lower_line for pattern in ignore_patterns):
            return

        try:
            os.write(self._saved_stderr_fd, line_bytes)
        except OSError:
            return
