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
    """goal_dim > 0 なら「指示」（例：右眼／左眼の one-hot）を特徴点の座標につなげて入れる（M4-F）。"""

    def __init__(self, keypoints=16, goal_dim=0):
        super().__init__()
        self.goal_dim = goal_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),   # 96 → 48
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, keypoints, 1),
        )
        self.keypoints = SpatialSoftmax()
        self.head = nn.Sequential(
            nn.Linear(2 * keypoints + goal_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, img, goal=None):
        """img: (n, 3, 96, 96) float 0〜1、goal: (n, goal_dim) → 指令 (n, 2)、MAX_STEP_MM で割った値（-1〜1 付近）。"""
        feat = self.keypoints(self.conv(img))
        if self.goal_dim:
            feat = torch.cat([feat, goal], dim=1)
        return self.head(feat)


def goal_onehot(goal, n=2):
    """指示の番号（0=右眼、1=左眼）→ one-hot (len, n) の float テンソル。"""
    g = torch.as_tensor(np.atleast_1d(goal), dtype=torch.long)
    return F.one_hot(g, n).float()


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
        # 指示つきの生徒（M4-F）は、env から「指示」（env.goal：0=右眼、1=左眼）だけを受け取る。目標の位置は見ない
        goal = goal_onehot(env.goal) if self.net.goal_dim else None
        out = self.net(to_tensor(obs[None]), goal)[0].numpy() * MAX_STEP_MM
        return np.clip(out, -MAX_STEP_MM, MAX_STEP_MM)
