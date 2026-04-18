from pathlib import Path
import torch
import torch.nn as nn


FRAME_SKIP = 1         # frame
WINDOW_SIZE = 15       # frames per inference window (matches demo inference / original training)
FAKE_THRESHOLD = 0.85  # window-mean fake-prob threshold

_cvit_instance = None


# CViT2 model architecture
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


def get_cvit_detector(device):
    global _cvit_instance
    if _cvit_instance is None:
        _cvit_instance = CViTDetector(device)
    return _cvit_instance


class CViTDetector(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = CViT(image_size=224, patch_size=7, num_classes=2, channels=512, dim=1024, depth=6, heads=8, mlp_dim=2048)
        weights_path = Path(__file__).resolve().parents[1] / "weights" / "cvit2_deepfake_detection_ep_50.pth"
        self._load_weights(weights_path, device)

    def _load_weights(self, weights_path, device):
        checkpoint = torch.load(str(weights_path), map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

    def forward(self, img):
        return self.model(img)

    def predict_single(self, face_tensor: torch.Tensor):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(face_tensor.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)
        return prob[0, 0].item()

    def predict_window(self, face_tensors: list[torch.Tensor]):
        if not face_tensors:
            return None
        device = next(self.model.parameters()).device
        batch = torch.stack(face_tensors).to(device)
        with torch.no_grad():
            logits = self.model(batch)   # [N, 2]
            probs = torch.sigmoid(logits)  # [N, 2]
        return probs[:, 0].mean().item()  # mean fake-prob
