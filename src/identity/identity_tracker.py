import cv2
import numpy as np
import threading
from dataclasses import dataclass
from insightface.app import FaceAnalysis


@dataclass
class IdentityResult:
    similarity: float | None = None
    avg_similarity: float | None = None
    min_similarity: float | None = None
    drift: float | None = None
    identity_score: float | None = None
    frame_count: int = 0
    embedding_count: int = 0


class IdentityTracker:
    def __init__(self, providers=None):
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        app = FaceAnalysis(name='buffalo_l', providers=providers)
        app.prepare(ctx_id=0, det_size=(160, 160))

        self.rec_model = None
        for model in app.models.values():
            if hasattr(model, 'get_feat'):
                self.rec_model = model
                break

        if self.rec_model is None:
            raise RuntimeError("ArcFace recognition model not found in InsightFace model pack")

        self._input_size = getattr(self.rec_model, 'input_size', (112, 112))
        self.reference_embedding = None
        self._similarities: list[float] = []
        self._lock = threading.Lock()

    def process(self, aligned_face_pil, frame_count):
        if aligned_face_pil is None:
            return None

        embedding = self._compute_embedding(aligned_face_pil)
        if embedding is None:
            return None

        with self._lock:
            if self.reference_embedding is None:
                self.reference_embedding = embedding.copy()
                self._similarities.append(1.0)
                return IdentityResult(
                    similarity=1.0,
                    avg_similarity=1.0,
                    min_similarity=1.0,
                    drift=0.0,
                    identity_score=1.0,
                    frame_count=frame_count,
                    embedding_count=1,
                )

            similarity = float(np.dot(embedding, self.reference_embedding))
            self._similarities.append(similarity)

            avg_sim = float(np.mean(self._similarities))
            min_sim = float(np.min(self._similarities))
            drift = float(np.std(self._similarities)) if len(self._similarities) > 1 else 0.0
            identity_score = self._compute_identity_score(avg_sim, drift)

            return IdentityResult(
                similarity=similarity,
                avg_similarity=avg_sim,
                min_similarity=min_sim,
                drift=drift,
                identity_score=identity_score,
                frame_count=frame_count,
                embedding_count=len(self._similarities),
            )

    def _compute_embedding(self, aligned_face_pil):
        face_np = np.array(aligned_face_pil)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        face_resized = cv2.resize(face_bgr, self._input_size)

        embedding = self.rec_model.get_feat(face_resized).flatten()
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None
        return embedding / norm

    @staticmethod
    def _compute_identity_score(avg_similarity, drift):
        sim_component = max(0.0, min(1.0, (avg_similarity - 0.2) / 0.6))
        drift_penalty = min(1.0, drift / 0.15)
        return sim_component * (1.0 - 0.3 * drift_penalty)

    def reset(self):
        with self._lock:
            self.reference_embedding = None
            self._similarities.clear()
