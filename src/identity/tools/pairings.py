"""Per-subject pairing recommendations from cached DFM/source embeddings with gender filter."""
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.utils import face_align

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from identity.tools.metadata import DFM_IDENTITIES, SOURCE_IDENTITIES, age_today
from identity.tools.similarity import (
    VIDEO_POSE_DEG, _frontal, _has_hand, _largest,
    categorize, cosine, embed_video, load_app,
)


SRC_DIR = Path(__file__).resolve().parent / "sources"
TGT_DIR = Path(r"C:\Users\daviddrtil\docs\school\ing\thesis\recordings\targets")
DFM_DIR = Path(r"C:\Users\daviddrtil\docs\repos\DeepFaceLive_NVIDIA_build_07_09_2023\DeepFaceLive_NVIDIA\userdata\dfm_models")
CACHE_PATH = Path(__file__).resolve().parent / "embeddings_cache.npz"
PREVIEWS_DIR = Path(__file__).resolve().parent / "previews"
OUT_DIR = Path(__file__).resolve().parent

SUBJECTS = {
    'subject1_male': [
        TGT_DIR / "01_head_rotation_old_camera.mp4",
        TGT_DIR / "real1.mp4",
        TGT_DIR / "03_cover_eye.mp4",
    ],
    'subject2_female': [TGT_DIR / "anna_berkova" / "02_glasses_off_extra_action.mp4"],
}
SUBJECT1_REAL = SUBJECTS['subject1_male']

GENDER_LABEL = {0: 'female', 1: 'male'}
TOP_N = 3
SUBJECT_GENDER_RE = re.compile(r'_(male|female)$')


def _resolve_size(shape, fallback=224):
    h = shape[1] if isinstance(shape[1], int) else fallback
    w = shape[2] if isinstance(shape[2], int) else fallback
    return w, h


def _swap_face(sess, aligned_face_bgr):
    inp = sess.get_inputs()[0]
    w, h = _resolve_size(inp.shape)
    resized = cv2.resize(aligned_face_bgr, (w, h)) if (w, h) != aligned_face_bgr.shape[:2][::-1] else aligned_face_bgr
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for o in sess.run(None, {inp.name: rgb[None]}):
        a = np.asarray(o)
        if a.ndim == 4 and a.shape[3] == 3:
            return cv2.cvtColor((np.clip(a[0], 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return None


def _rec_model(app):
    for m in app.models.values():
        if hasattr(m, 'get_feat'):
            return m
    return None


def aligned_face_from_frame(app, frame_bgr):
    face = _largest(app.get(frame_bgr))
    if face is None:
        return None
    return face_align.norm_crop(frame_bgr, face.kps, image_size=224)


def dfm_embedding(app, dfm_path, aligned_face_bgr):
    rec = _rec_model(app)
    if rec is None:
        return None, 'no rec model'
    try:
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(str(dfm_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except Exception as e:
        return None, f'cannot load: {type(e).__name__}'
    swap_bgr = _swap_face(sess, aligned_face_bgr)
    if swap_bgr is None:
        return None, 'no swap output'
    feat = rec.get_feat(cv2.resize(swap_bgr, (112, 112))).flatten()
    norm = np.linalg.norm(feat)
    return ((feat / norm), None) if norm > 1e-6 else (None, 'zero embedding')


def neutral_aligned_face(app, video_path, samples=30, max_deg=VIDEO_POSE_DEG):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    for idx in np.linspace(0, total - 1, num=min(samples, total), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or _has_hand(frame):
            continue
        face = _largest(app.get(frame))
        if face is None or not _frontal(face, max_deg):
            continue
        cap.release()
        return face_align.norm_crop(frame, face.kps, image_size=224)
    cap.release()
    return None


def load_cache():
    if not CACHE_PATH.exists():
        sys.exit(f"Cache not found: {CACHE_PATH}. Run build_cache.py first.")
    data = np.load(CACHE_PATH, allow_pickle=False)
    return {
        'kinds': data['kinds'].tolist(),
        'paths': data['paths'].tolist(),
        'labels': data['labels'].tolist(),
        'embeddings': data['embeddings'],
        'genders': data['genders'].tolist(),
        'ages': data['ages'].tolist(),
    }


def subject_embedding(app, video_paths):
    embs = []
    for vp in video_paths:
        emb, _ = embed_video(app, vp)
        if emb is not None:
            embs.append(emb)
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm > 1e-6 else None


def gender_from_subject_id(subject_id):
    m = SUBJECT_GENDER_RE.search(subject_id or '')
    if not m:
        return None
    return 1 if m.group(1) == 'male' else 0


def rank(cache, subject_emb, subject_gender, kind):
    rows = []
    for i in range(len(cache['paths'])):
        if cache['kinds'][i] != kind:
            continue
        if subject_gender is not None and cache['genders'][i] != subject_gender:
            continue
        rows.append({
            'path': cache['paths'][i],
            'label': cache['labels'][i],
            'gender': cache['genders'][i],
            'age': cache['ages'][i],
            'sim': float(np.dot(subject_emb, cache['embeddings'][i])),
        })
    rows.sort(key=lambda r: -r['sim'])
    return rows


def _enrich_row(r, kind):
    meta = (DFM_IDENTITIES if kind == 'dfm' else SOURCE_IDENTITIES).get(r['label'], {})
    name = meta.get('name', r['label'])
    age = age_today(meta.get('birth_year')) if kind == 'dfm' else None
    return name, age


def _table(rows, kind):
    out = ["| Rank | File | Identity | Cosine | Category | Gender | Age |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        name, age = _enrich_row(r, kind)
        age_str = '?' if age is None else age
        out.append(f"| {i} | `{r['label']}` | {name} | {r['sim']:.4f} | {categorize(r['sim'])} | {GENDER_LABEL.get(r['gender'], '?')} | {age_str} |")
    return "\n".join(out)


def _preview_path(label, kind, source_path):
    if kind == 'dfm':
        for ext in ('.jpg', '.png'):
            p = PREVIEWS_DIR / f"{label}{ext}"
            if p.exists():
                return p.relative_to(OUT_DIR).as_posix()
        return None
    return Path(source_path).as_posix()


def _previews_block(rows, kind, width=240):
    cells = []
    for r in rows[:TOP_N]:
        name, age = _enrich_row(r, kind)
        cells.append((name, _preview_path(r['label'], kind, r['path']), r['sim'], age))
    if not cells:
        return ""
    headers = " | ".join(f"#{i + 1}" for i in range(len(cells)))
    sep = " | ".join("---" for _ in cells)
    images = " | ".join(
        f'<img src="{rel}" width="{width}" alt="{name}"/>' if rel else '_(no preview)_'
        for name, rel, _, _ in cells
    )
    captions = " | ".join(
        f"**{name}**<br/>cosine {sim:.4f}" + (f", age {age}" if age is not None else "")
        for name, _, sim, age in cells
    )
    return "\n".join([f"| {headers} |", f"| {sep} |", f"| {images} |", f"| {captions} |", ""])


def write_subject_report(subject_id, gender, ranked_dfm, ranked_src):
    md = [
        f"# Pairing recommendations for `{subject_id}`",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')}. Subject gender from session_id: **{GENDER_LABEL.get(gender, 'unknown')}**. Candidates filtered to same gender (manual labels from metadata.py)._",
        "",
        "## We will swap you onto these top 3 identities",
        "",
        "### DFM swap models (DeepFaceLive)",
        "",
        _previews_block(ranked_dfm, 'dfm'),
        "## Full ranking — DFM models",
        "",
        _table(ranked_dfm, 'dfm'),
        "",
        "### Source images (FaceFusion / inswapper)",
        "",
        _previews_block(ranked_src, 'source'),
        "## Full ranking — source images",
        "",
        _table(ranked_src, 'source'),
        "",
    ]
    out = OUT_DIR / f"recommendations_{subject_id}.md"
    out.write_text("\n".join(md), encoding='utf-8')
    print(f"Wrote {out}")


if __name__ == '__main__':
    app = load_app()
    cache = load_cache()
    for subject_id, video_paths in SUBJECTS.items():
        emb = subject_embedding(app, video_paths)
        if emb is None:
            print(f"{subject_id}: no usable subject embedding")
            continue
        gender = gender_from_subject_id(subject_id)
        if gender is None:
            print(f"{subject_id}: no _male/_female suffix in subject id, skipping")
            continue
        ranked_dfm = rank(cache, emb, gender, kind='dfm')
        ranked_src = rank(cache, emb, gender, kind='source')
        write_subject_report(subject_id, gender, ranked_dfm, ranked_src)
