"""生徒AIの「脳」：細隙灯カメラの画像 → 指令 (Δx, Δz)[mm] を出す小さなCNN。

構成：畳み込み層で特徴（瞳孔の縁・虹彩など）を見つける → 空間ソフトマックスで
「その特徴が画像のどこにあるか」の座標を取り出す → 全結合層で指令に変換する。
空間ソフトマックス（Levine et al., 2016）は位置を画素以下の細かさで出せるので、
1mm ≒ 2画素 の精度が要るこの課題に向いている。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.env import MAX_STEP_MM


class SpatialSoftmax(nn.Module):
    """各チャンネルの反応の「重心」を (x, y) 座標（-1〜1）として出す。"""

    def forward(self, x):
        n, c, h, w = x.shape
        p = F.softmax(x.flatten(2), dim=-1).view(n, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ex = (p.sum(2) * xs).sum(-1)  # (n, c)
        ey = (p.sum(3) * ys).sum(-1)
        return torch.cat([ex, ey], dim=1)  # (n, 2c)


class StudentNet(nn.Module):
    def __init__(self, keypoints=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),   # 96 → 48
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, keypoints, 1),
        )
        self.keypoints = SpatialSoftmax()
        self.head = nn.Sequential(
            nn.Linear(2 * keypoints, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, img):
        """img: (n, 3, 96, 96) float 0〜1 → 指令 (n, 2)、MAX_STEP_MM で割った値（-1〜1 付近）。"""
        return self.head(self.keypoints(self.conv(img)))


def to_tensor(images):
    """uint8 (n, H, W, 3) → float (n, 3, H, W)。"""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0


class StudentPolicy:
    """学習済みの生徒。環境では policy(obs, env) として使う（env は見ない＝画像だけで決める）。"""

    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.net = StudentNet(**ckpt.get("config", {}))
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

    @torch.no_grad()
    def __call__(self, obs, env=None):
        out = self.net(to_tensor(obs[None]))[0].numpy() * MAX_STEP_MM
        return np.clip(out, -MAX_STEP_MM, MAX_STEP_MM)
