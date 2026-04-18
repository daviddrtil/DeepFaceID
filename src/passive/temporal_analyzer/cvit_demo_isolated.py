"""
CViT2 Deepfake Detection Demo (Isolated)

Preprocessing matches the original pred_func.py pipeline:
  - MediaPipe FaceDetection for bounding-box crop (replaces dlib/face_recognition)
  - 10 px padding, INTER_AREA resize to 224x224
  - float32 / 255, ImageNet normalize
  - Batch sigmoid inference, mean over frames, argmax (real_or_fake logic)

Optimisation: every 16th frame is sampled (no overlap) and the whole sequence
is fed to the model in a single forward pass.
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# ANSI colours (disabled on non-TTY / Windows without VT support)
_ANSI   = hasattr(os, "get_terminal_size") and os.isatty(1)
_RED    = "\033[91m" if _ANSI else ""
_YELLOW = "\033[93m" if _ANSI else ""
_RESET  = "\033[0m"  if _ANSI else ""


def _fake_colour(fake_prob: float) -> str:
    """Return the ANSI prefix for a fake-probability value."""
    if fake_prob > 0.9:
        return _RED
    if fake_prob > 0.8:
        return _YELLOW
    return ""


# ---------------------------------------------------------------------------
# CViT2 model (self-contained, no einops dependency)
# ---------------------------------------------------------------------------

class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kw):
        return self.fn(x, **kw) + x

class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kw):
        return self.fn(self.norm(x), **kw)

class _FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
    def forward(self, x):
        return self.net(x)

class _Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        d = x.shape[-1] // h
        qkv = self.to_qkv(x).reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().reshape(b, n, -1)
        return self.to_out(out)

class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                _Residual(_PreNorm(dim, _Attention(dim, heads=heads))),
                _Residual(_PreNorm(dim, _FeedForward(dim, mlp_dim))),
            ]))
    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class CViT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, num_classes=2, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = _Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, num_classes))

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.features(img)
        b, c, H, W = x.shape
        h, w = H // p, W // p
        y = x.reshape(b, c, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(b, h * w, p * p * c)
        y = self.patch_to_embedding(y)
        cls_tokens = self.cls_token.expand(y.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), dim=1)
        x += self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceDetector:
    """Bounding-box face crop via MediaPipe FaceDetection.

    Replicates the original face_recognition / dlib approach from pred_func.py:
      1. Detect the largest face in the frame.
      2. Expand the bbox by `padding` pixels on each side.
      3. Crop and resize to 224×224 with INTER_AREA.
    Returns a uint8 RGB numpy array (224, 224, 3), or None if no face found.
    """

    def __init__(self, padding: int = 10, min_confidence: float = 0.5):
        self.padding = padding
        # model_selection=1: full-range model suited for videos (up to 5 m)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_confidence,
        )

    def crop_face(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        result = self.detector.process(frame_rgb)
        if not result.detections:
            return None

        det = result.detections[0]
        bb  = det.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w) - self.padding)
        y1 = max(0, int(bb.ymin * h) - self.padding)
        x2 = min(w, int((bb.xmin + bb.width)  * w) + self.padding)
        y2 = min(h, int((bb.ymin + bb.height) * h) + self.padding)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


def preprocess_face(face_rgb: np.ndarray) -> torch.Tensor:
    """Replicate pred_func.preprocess_frame exactly.

    Input : uint8 RGB (224, 224, 3)
    Output: float32 CHW tensor, ImageNet-normalised
    """
    x = face_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))


def load_cvit_model(weights_path, device):
    """Load CViT2 model with pretrained weights."""
    model = CViT(
        image_size=224, patch_size=7, num_classes=2, channels=512,
        dim=1024, depth=6, heads=8, mlp_dim=2048,
    )
    checkpoint = torch.load(str(weights_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def extract_frames(video_path, skip):
    """Read video and return every `skip`-th frame (no overlap)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total / fps
    print(f"  Video info: {total} frames, {fps:.1f} FPS, {duration:.1f}s")

    frames, indices = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frames.append(frame)
            indices.append(idx)
        idx += 1
    cap.release()
    print(f"  Sampled {len(frames)} frames (every {skip}th from {idx} total)")
    return frames, indices


def process_video(video_path, model, detector, device, skip):
    """Run full CViT2 deepfake detection pipeline on one video."""
    print(f"\n{'=' * 60}")
    print(f"  Video: {video_path.name}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # --- frame extraction ---
    frames, indices = extract_frames(video_path, skip)
    if not frames:
        print("  ERROR: No frames extracted!")
        return None

    # --- face detection & crop (original bbox approach) ---
    print(f"  Detecting faces (MediaPipe bbox)...")
    face_tensors, face_indices, failed = [], [], 0
    t_face = time.time()

    for i, (frame, frame_idx) in enumerate(zip(frames, indices)):
        face_rgb = detector.crop_face(frame)
        if face_rgb is not None:
            face_tensors.append(preprocess_face(face_rgb))
            face_indices.append(frame_idx)
        else:
            failed += 1
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            print(f"    {i + 1}/{len(frames)} frames | {len(face_tensors)} faces, {failed} missed")

    face_time = time.time() - t_face
    print(f"  Face extraction: {face_time:.2f}s "
          f"({len(frames) / max(face_time, 1e-6):.1f} frames/s)")

    if not face_tensors:
        print("  ERROR: No faces detected in any frame!")
        return None

    # --- CViT2 batch inference (matches pred_vid in pred_func.py) ---
    batch = torch.stack(face_tensors).to(device)
    print(f"  Running CViT2 inference on {batch.shape[0]} face crops...")

    t_inf = time.time()
    with torch.no_grad():
        logits = model(batch)          # [N, 2]
        probs  = torch.sigmoid(logits) # [N, 2]
    inf_time = time.time() - t_inf

    # --- aggregate: replicate max_prediction_value + real_or_fake ---
    # class 0 = FAKE, class 1 = REAL  (original XOR: pred^1 == 1 => "FAKE")
    mean_probs = probs.mean(dim=0)
    pred_cls = torch.argmax(mean_probs).item()
    label  = "REAL" if pred_cls == 1 else "FAKE"
    fake_p = mean_probs[0].item()
    real_p = mean_probs[1].item()

    print(f"  Inference: {inf_time:.3f}s "
          f"({len(face_tensors) / max(inf_time, 1e-6):.1f} frames/s)")

    # --- per-frame predictions (all frames) ---
    n = len(face_tensors)
    print(f"\n  Per-frame predictions ({n} frames):")
    for i in range(n):
        fp, rp = probs[i, 0].item(), probs[i, 1].item()
        verdict = "REAL" if rp > fp else "FAKE"
        col = _fake_colour(fp)
        print(f"    Frame {face_indices[i]:5d}:  {col}FAKE={fp:.4f}{_RESET}  REAL={rp:.4f}  -> {col}{verdict}{_RESET}")

    total_time = time.time() - t0

    col = _fake_colour(fake_p)
    w = 39
    def _row(text, colour=""):
        return f"  | {colour}{text:{w - 4}s}{_RESET if colour else ''} |"
    print(f"\n  +{'-' * (w - 2)}+")
    print(_row(f"Prediction : {label}", col))
    print(_row(f"FAKE prob  : {fake_p:.4f}", col))
    print(_row(f"REAL prob  : {real_p:.4f}"))
    print(_row(f"Frames used: {n}"))
    print(_row(f"Total time : {total_time:.2f}s"))
    print(f"  +{'-' * (w - 2)}+")

    return dict(video=video_path.name, label=label,
                fake_prob=fake_p, real_prob=real_p,
                num_frames=n, time=total_time)


def resolve_default_paths():
    demo_dir = Path(__file__).resolve().parent
    repo_dir = demo_dir.parent
    return {
        "weights": repo_dir / "weight" / "cvit2_deepfake_detection_ep_50.pth",
        "inputs_dir": demo_dir / "inputs",
    }


def parse_args():
    paths = resolve_default_paths()
    p = argparse.ArgumentParser(description="CViT2 deepfake detection demo (isolated).")
    p.add_argument(
        "--videos", type=Path, nargs="+",
        default=[
            paths["inputs_dir"] / "daviddrtil.mp4",
            paths["inputs_dir"] / "tomcruise.mp4",
        ],
        help="Input video path(s).",
    )
    p.add_argument("--weights", type=Path, default=paths["weights"],
                   help="CViT2 checkpoint path.")
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                   help="Inference device.")
    p.add_argument("--skip", type=int, default=8,
                   help="Take every N-th frame (default: 8).")
    p.add_argument("--padding", type=int, default=10,
                   help="Face bbox padding in pixels (default: 10).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    for v in args.videos:
        if not v.exists():
            raise FileNotFoundError(f"Video not found: {v}")

    print(f"Loading CViT2 model from {args.weights.name} ...")
    model = load_cvit_model(args.weights, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    print(f"Initializing MediaPipe face detector ...")
    detector = FaceDetector(padding=args.padding)

    results = []
    for vp in args.videos:
        r = process_video(vp, model, detector, device, args.skip)
        if r:
            results.append(r)

    if results:
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY")
        print(f"{'=' * 60}")
        for r in results:
            print(f"  {r['video']:30s} -> {r['label']:4s}  "
                  f"(fake={r['fake_prob']:.4f}, real={r['real_prob']:.4f}, "
                  f"{r['num_frames']} frames, {r['time']:.2f}s)")
