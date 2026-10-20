"""お手本データ（ep_XXXXX.npz）をまとめて読み込む。M2 の先生のデータと、DAgger で追加したデータの両方に使う。"""
from pathlib import Path

import numpy as np


def load_episodes(dirs):
    """複数フォルダの ep_*.npz を読み、エピソードごとのリストを返す。
    指示（goal：0=右眼、1=左眼、M4-F）は、ファイルにあればそれを、無ければフォルダ名から決める
    （`_L` で終わるフォルダ＝左眼のお手本、それ以外＝右眼）。"""
    eps = []
    for d in dirs:
        default_goal = 1 if Path(d).name.endswith("_L") else 0
        for f in sorted(Path(d).glob("ep_*.npz")):
            z = np.load(f)
            eps.append({"seed": int(f.stem.split("_")[1]), "images": z["images"],
                        "actions": z["actions"], "errors": z["errors"],
                        "goal": int(z["goal"]) if "goal" in z else default_goal})
    return eps


def split_and_stack(eps, val_every=10):
    """エピソード単位で学習用・確認用に分ける（seed が val_every の倍数のものを確認用に）。
    同じエピソードの画像は互いによく似ているので、画像単位で分けると確認用の成績が甘くなる。
    返り値：(画像, 指令, ずれ, 指示) の組を、学習用・確認用それぞれ。"""
    def stack(sel):
        return (np.concatenate([e["images"] for e in sel]),
                np.concatenate([e["actions"] for e in sel]),
                np.concatenate([e["errors"] for e in sel]),
                np.concatenate([np.full(len(e["actions"]), e["goal"]) for e in sel]))
    train = [e for e in eps if e["seed"] % val_every != 0]
    val = [e for e in eps if e["seed"] % val_every == 0]
    return stack(train), stack(val)
