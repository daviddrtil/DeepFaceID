"""ArcFace cosine similarity. Usage: python src/identity/tools/similarity.py source target."""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis


IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
VIDEO_POSE_DEG = 15.0
VIDEO_SAMPLES = 12
VIDEO_KEEP = 5

_hands = None


def load_app(providers=('CUDAExecutionProvider', 'CPUExecutionProvider')):
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)
    with contextlib.redirect_stdout(io.StringIO()):
        app = FaceAnalysis(name='buffalo_l', providers=list(providers))
        app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def list_inputs(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT | VIDEO_EXT)


def is_video(p):
    return Path(p).suffix.lower() in VIDEO_EXT


def _largest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


def _frontal(face, max_deg):
    pose = getattr(face, 'pose', None)
    return pose is None or all(abs(float(v)) <= max_deg for v in pose[:3])


def _has_hand(image_bgr):
    global _hands
    if _hands is None:
        _hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return bool(_hands.process(rgb).multi_hand_landmarks)


def _normalize(vectors):
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm > 1e-6 else None


def embed_image(app, image_bgr):
    face = _largest(app.get(image_bgr))
    if face is None:
        return None, 'no face'
    return face.normed_embedding, None


def embed_video(app, path, max_deg=VIDEO_POSE_DEG, samples=VIDEO_SAMPLES, keep=VIDEO_KEEP):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 'empty video'
    embs = []
    for idx in np.linspace(0, total - 1, num=min(samples, total), dtype=int):
        if len(embs) >= keep:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None or _has_hand(frame):
            continue
        face = _largest(app.get(frame))
        if face is None or not _frontal(face, max_deg):
            continue
        embs.append(face.normed_embedding)
    cap.release()
    out = _normalize(embs)
    return (out, None) if out is not None else (None, 'no usable frame')


def embed(app, path):
    if is_video(path):
        return embed_video(app, path)
    img = cv2.imread(str(path))
    if img is None:
        return None, 'unreadable'
    return embed_image(app, img)


def aggregate(app, paths):
    embs, log = [], []
    for p in paths:
        e, err = embed(app, p)
        log.append(f'{p.name}: {"ok" if e is not None else err}')
        if e is not None:
            embs.append(e)
    return _normalize(embs), log


def cosine(a, b):
    return float(np.dot(a, b))


def categorize(s):
    if s >= 0.7: return 'very close'
    if s >= 0.5: return 'close'
    if s >= 0.3: return 'distant'
    return 'far'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    app = load_app()
    src, src_log = aggregate(app, list_inputs(args.source))
    tgt, tgt_log = aggregate(app, list_inputs(args.target))
    for ln in src_log + tgt_log:
        print(ln)
    if src is None or tgt is None:
        sys.exit('no usable embeddings')
    sim = cosine(src, tgt)
    above = 'above' if sim >= args.threshold else 'below'
    print(f'\ncosine = {sim:.4f}  {categorize(sim)}  {above} {args.threshold:.2f}')
