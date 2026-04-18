from pathlib import Path
import torch
import torch.nn as nn
from passive.spatial_analyzer.xception import Xception

_ucf_instance = None


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )
        self.do = nn.Dropout(0.2)

    def forward(self, x):
        bs = x.size(0)
        x_feat = self.pool(x).view(bs, -1)
        logits = self.do(self.mlp(x_feat))
        return logits, x_feat


def get_ucf_detector(device):
    global _ucf_instance
    if _ucf_instance is None:
        _ucf_instance = UCFDetector(device)
    return _ucf_instance


class UCFDetector(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder_f = Xception()
        self.block_sha = Conv2d1x1(in_f=512, hidden_dim=256, out_f=256)
        self.head_sha = Head(in_f=256, hidden_dim=512, out_f=2)

        weights_path = Path(__file__).resolve().parents[1] / "weights" / "ucf_best.pth"
        self._load_weights(weights_path, device)

    @staticmethod
    def _extract_checkpoint_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            if "net" in checkpoint:
                return checkpoint["net"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

    @staticmethod
    def _clean_checkpoint_keys(checkpoint):
        cleaned = {}
        for key, value in checkpoint.items():
            prefix = "module."
            key = key[len(prefix):] if key.startswith(prefix) else key
            if key.startswith(("encoder_f.", "block_sha.", "head_sha.")):
                cleaned[key] = value
        return cleaned

    def _load_weights(self, weights_path, device):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        checkpoint = self._extract_checkpoint_state_dict(checkpoint)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint format is unsupported.")

        cleaned = self._clean_checkpoint_keys(checkpoint)
        if not cleaned:
            raise RuntimeError("No compatible weights found for isolated UCF model.")

        load_result = self.load_state_dict(cleaned, strict=False)
        if not cleaned or len(load_result.missing_keys) == len(self.state_dict()):
            raise RuntimeError("Failed to load isolated UCF weights.")
        if load_result.unexpected_keys:
            print(f"Skipped {len(load_result.unexpected_keys)} unrelated checkpoint keys.")
            print(load_result.unexpected_keys)
        if load_result.missing_keys:
            print(f"Missing {len(load_result.missing_keys)} model keys (not needed for demo path).")
            print(load_result.missing_keys)

        self.to(device)
        self.eval()

    def forward(self, x):
        forgery_features = self.encoder_f.features(x)
        f_share = self.block_sha(forgery_features)
        logits, _ = self.head_sha(f_share)
        return torch.softmax(logits, dim=1)[:, 1]

    def predict(self, input_tensor):
        device = next(self.parameters()).device
        with torch.no_grad():
            probability = self(input_tensor.to(device))
        if probability.numel() != 1:
            raise ValueError("UCFDetector.predict expects a single-sample batch.")
        return probability.item()
