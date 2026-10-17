"""M4-A：瞳孔検出の精度を、お手本データ（本当のずれ errors が記録されている）で確かめる。

出力:
    検出誤差の平均・中央値・95%点・最大、見つからなかった割合、取り違え（誤差3mm以上）の割合
    outputs/m4a_detect_failures.png … 誤差の大きい画像の一覧（緑の円＝検出、赤の十字＝本当の瞳孔中心）

実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m4a_detect_check.py --data data/demos data/demos_L
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from policies.modular import PX_PER_MM, detect_pupil, pupil_offset_mm

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/demos"])
    p.add_argument("--n", type=int, default=4000, help="調べる画像の数（ランダムに選ぶ）")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    files = [f for d in args.data for f in sorted((ROOT / d).glob("ep_*.npz"))]
    picks = []
    for f in rng.choice(files, args.n):
        z = np.load(f)
        t = rng.integers(len(z["images"]))
        picks.append((z["images"][t], z["errors"][t]))

    errs, misses, worst = [], 0, []
    for img, true in picks:
        e = pupil_offset_mm(img)
        if e is None:
            misses += 1
            worst.append((np.inf, img, true))
            continue
        d = float(np.linalg.norm(e - true))
        errs.append(d)
        worst.append((d, img, true))
    errs = np.array(errs)
    print(f"images {args.n}  missed {misses} ({100 * misses / args.n:.1f}%)")
    signed = np.array([pupil_offset_mm(img) - true for img, true in picks if pupil_offset_mm(img) is not None])
    print(f"bias (detected - true) mean x {signed[:, 0].mean():+.3f}  z {signed[:, 1].mean():+.3f} mm"
          f"  → 校正値 CALIB_OFFSET_MM の候補（右眼のデータだけで測ること）")
    print(f"error mm: mean {errs.mean():.3f}  median {np.median(errs):.3f}  p95 {np.percentile(errs, 95):.3f}  "
          f"max {errs.max():.2f}  wrong(>=3mm) {(errs >= 3).sum()} ({100 * (errs >= 3).mean():.1f}%)")

    # 誤差の大きい順に24枚を一覧にする
    worst.sort(key=lambda w: -w[0])
    tiles = []
    for d, img, true in worst[:24]:
        big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        c = (big.shape[1] - 1) / 2
        tx, ty = c + true[0] * PX_PER_MM * 3, c - true[1] * PX_PER_MM * 3
        cv2.drawMarker(big, (int(tx), int(ty)), (255, 0, 0), cv2.MARKER_CROSS, 12, 1)
        found = detect_pupil(img)
        if found:
            cv2.circle(big, (int(found[0] * 3), int(found[1] * 3)), int(found[2] * 3), (0, 255, 0), 1)
        cv2.putText(big, "miss" if np.isinf(d) else f"{d:.1f}mm", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        tiles.append(big)
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    out = ROOT / "outputs" / "m4a_detect_failures.png"
    imageio.imwrite(out, np.vstack(rows))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
