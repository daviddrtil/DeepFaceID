import cv2
import numpy as np
import queue
import threading
from dataclasses import dataclass
from insightface.app import FaceAnalysis
from preprocessing.live_video_queue import LiveVideoQueue


@dataclass
class IdentityResult:
    similarity: float | None = None
    avg_similarity: float | None = None
    min_similarity: float | None = None
    drift: float | None = None
    identity_score: float | None = None
    frame_count: int = 0
    embedding_count: int = 0


_rec_model = None
_rec_input_size = None


def _load_rec_model(providers=None):
    global _rec_model, _rec_input_size
    if _rec_model is not None:
        return

    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(160, 160))

    for model in app.models.values():
        if hasattr(model, 'get_feat'):
            _rec_model = model
            _rec_input_size = getattr(model, 'input_size', (112, 112))
            return

    raise RuntimeError("ArcFace recognition model not found in InsightFace model pack")


class IdentityTracker:
    def __init__(self, providers=None):
        _load_rec_model(providers)
        self.reference_embedding = None
        self._similarities: list[float] = []
        self._lock = threading.Lock()
        self._input_queue = LiveVideoQueue(maxsize=2)
        self._latest_result = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                aligned_face_pil, frame_count = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = self._process(aligned_face_pil, frame_count)
            if result is not None:
                with self._lock:
                    self._latest_result = result

    def submit(self, aligned_face_pil, frame_count):
        if aligned_face_pil is None:
            return
        self._input_queue.put_latest((aligned_face_pil, frame_count))

    def get_result(self):
        with self._lock:
            return self._latest_result

    def _process(self, aligned_face_pil, frame_count):
        embedding = self._compute_embedding(aligned_face_pil)
        if embedding is None:
            return None

        with self._lock:
            if self.reference_embedding is None:
                self.reference_embedding = embedding.copy()

            similarity = float(np.dot(embedding, self.reference_embedding))
            self._similarities.append(similarity)
            avg_sim = float(np.mean(self._similarities))
            min_sim = float(np.min(self._similarities))
            drift = float(np.std(self._similarities)) if len(self._similarities) > 1 else 0.0

            return IdentityResult(
                similarity=similarity,
                avg_similarity=avg_sim,
                min_similarity=min_sim,
                drift=drift,
                identity_score=self._compute_identity_score(avg_sim, drift),
                frame_count=frame_count,
                embedding_count=len(self._similarities),
            )

    def _compute_embedding(self, aligned_face_pil):
        face_np = np.array(aligned_face_pil)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        face_resized = cv2.resize(face_bgr, _rec_input_size)

        embedding = _rec_model.get_feat(face_resized).flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None
        return embedding / norm

    @staticmethod
    def _compute_identity_score(avg_similarity, drift):
        sim_component = max(0.0, min(1.0, (avg_similarity - 0.2) / 0.6))
        drift_penalty = min(1.0, drift / 0.15)
        return sim_component * (1.0 - 0.3 * drift_penalty)

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reset(self):
        self.stop()
        self.reference_embedding = None
        self._similarities.clear()
        self._latest_result = None
        self._stop_event.clear()
        self._input_queue = LiveVideoQueue(maxsize=2)
        self._thread = None
