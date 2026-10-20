"""BC（行動クローニング）：画像 → 先生の指令 を真似るように生徒のCNNを学習する。

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python learning/train_bc.py
DAgger では --data に追加データのフォルダを足し、--init で前回の重みから続きを学習する。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learning.dataset import load_episodes, split_and_stack
from learning.model import StudentNet, goal_onehot, to_tensor
from sim.env import MAX_STEP_MM

ROOT = Path(__file__).resolve().parents[1]


def augment(x, g):
    """色・明るさ・コントラストを少し変え、ノイズを足す。フレームスタック（3×k チャンネル）でも、
    同じ組の画像には同じ変化をかける（M4-E）。"""
    n, ch, h, w = x.shape
    return _augment(x.view(n, ch // 3, 3, h, w), g).view(n, ch, h, w)


def _augment(x, g):
    """色・明るさ・コントラストを少し変え、ノイズを足す（位置は変えない＝正解の指令は変わらない）。"""
    n = x.shape[0]    # x: (n, 枚数, 3, H, W)
    bright = torch.empty(n, 1, 1, 1, 1).uniform_(0.75, 1.25, generator=g)
    gain = torch.empty(n, 1, 3, 1, 1).uniform_(0.9, 1.1, generator=g)
    mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
    contrast = torch.empty(n, 1, 1, 1, 1).uniform_(0.8, 1.2, generator=g)
    x = (x - mean) * contrast + mean
    x = x * bright * gain + torch.randn(x.shape, generator=g) * 0.02
    return x.clamp(0, 1)


def stacked(imgs, hist, idx):
    """番号 idx の画像に、直前の画像を色のチャンネル方向に重ねる（フレームスタック、M4-E）。"""
    return np.concatenate([imgs[hist[idx, j]] for j in range(hist.shape[1])], axis=-1)


@torch.no_grad()
def evaluate(net, imgs, acts, errs, goals, hist, bs=1024):
    net.eval()
    pred = np.concatenate([net(to_tensor(stacked(imgs, hist, np.arange(i, min(i + bs, len(imgs))))),
                               goal_onehot(goals[i:i + bs]) if net.goal_dim else None).numpy()
                           for i in range(0, len(imgs), bs)]) * MAX_STEP_MM
    net.train()
    ae = np.linalg.norm(pred - acts, axis=1)  # 指令の誤差の大きさ [mm]
    near = np.linalg.norm(errs, axis=1) < 2.0
    return {"val_err_mm": float(ae.mean()), "val_err_near_mm": float(ae[near].mean()),
            "val_err_far_mm": float(ae[~near].mean())}


def train(data, out, init=None, epochs=40, bs=256, lr=1e-3, flip=False, goal=False, stack=1):
    """data のフォルダ群で学習し、out に重みを、out.json に学習の経過を保存する。
    flip=True なら、半分の画像を左右反転し、正解の Δx の符号を逆にして混ぜる（右眼の見え方 → 左眼の見え方）。
    goal=True なら「指示」（0=右眼、1=左眼）も入力する生徒を学習する（M4-F）。"""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    (ti, ta, te, tg, th), (vi, va, ve, vg, vh) = split_and_stack(load_episodes(data), stack=stack)
    print(f"train {len(ti)} frames / val {len(vi)} frames", flush=True)

    config = {**({"goal_dim": 2} if goal else {}), **({"stack": stack} if stack > 1 else {})}
    net = StudentNet(**config)
    if init:
        net.load_state_dict(torch.load(init, map_location="cpu")["model"])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(ti) // bs))
    target = torch.from_numpy(ta / MAX_STEP_MM).float()

    history = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(ti), generator=g)
        losses = []
        for k in range(len(ti) // bs):
            idx = perm[k * bs:(k + 1) * bs].numpy()
            x = augment(to_tensor(stacked(ti, th, idx)), g)
            y = target[idx].clone()
            if flip:
                # 画像の右 = +x なので、左右反転したら Δx の符号も逆になる
                m = torch.rand(len(idx), generator=g) < 0.5
                x[m] = x[m].flip(-1)
                y[m, 0] = -y[m, 0]
            loss = F.mse_loss(net(x, goal_onehot(tg[idx]) if goal else None), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        h = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), **evaluate(net, vi, va, ve, vg, vh),
             "seconds": round(time.time() - t0)}
        history.append(h)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in h.items()), flush=True)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "config": config, "data": [str(d) for d in data]}, out)
    args = {"data": [str(d) for d in data], "init": init, "epochs": epochs, "bs": bs, "lr": lr, "flip": flip, "goal": goal, "stack": stack}
    out.with_suffix(".json").write_text(json.dumps({"args": args, "history": history}, indent=1))
    print("saved", out, flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "demos")])
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "bc.pt"))
    p.add_argument("--init", default=None, help="続きから学習するときの重み")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--flip", action="store_true", help="左右反転のデータ拡張（右眼のデータから左眼の見え方を作る）")
    p.add_argument("--goal", action="store_true", help="「指示」（右眼／左眼）も入力する生徒（M4-F）")
    p.add_argument("--stack", type=int, default=1, help="フレームスタックの枚数（今＋直前の画像、M4-E）")
    a = p.parse_args()
    train(a.data, a.out, a.init, a.epochs, a.bs, a.lr, a.flip, a.goal, a.stack)


if __name__ == "__main__":
    main()
