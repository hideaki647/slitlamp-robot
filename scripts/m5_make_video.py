"""M5：提出用の動画（約2分）を、これまでの動画とグラフをつないで作る。日本語の字幕・再生速度の表記つき。

画面：上 1280×600 に素材（縦横比を保って中央に置く）、下 120 画素の帯に「章の名前」と字幕。
素材の動画は実時間（50fps）で記録してあるので、ここで速さを変えて 30fps に並べ直す。
前提：outputs/ に M0〜M4 の動画・グラフがあること（各 scripts/m*_demo.py・m*_plot.py の出力）。
実行（slitlamp_robot フォルダで）:
    docker compose run --rm sim python scripts/m5_make_video.py
"""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
W, H, FPS = 1280, 720, 30
AREA_H = 600                       # 素材を置く高さ。残り 120 画素が字幕の帯
BG = (18, 18, 22)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"   # Dockerfile の fonts-noto-cjk


def font(size):
    return ImageFont.truetype(FONT, size)


# 動画の流れ。kind：card（文字だけの画面）／clip（動画）／image（グラフなど）
# captions：(この区切りの中での開始秒, 字幕)。speed：素材の再生速度（0.5＝スロー）
TIMELINE = [
    dict(kind="clip", src="m0_orbit.mp4", speed=1.0, chapter="",
         title=["細隙灯アライメントロボット", "ロボットアームがジョイスティックを握り、カメラ画像だけで瞳孔に合わせる"],
         captions=[(0, "物理シミュレータ MuJoCo で作った仮想の診察室。患者はすべて合成CG")]),
    dict(kind="clip", src="m1_demo.mp4", speed=2.0, end=20, chapter="システム",
         captions=[(0, "人の手のための操作レバー（ジョイスティック）を、ロボットアームがそのまま握って操作する"),
                   (5, "倒す → 本体が左右・前後に動く　／　回す → 上下に動く（既存の細隙灯を改造せずに遠隔化）")]),
    dict(kind="clip", src="m2_expert_demo.mp4", speed=0.5, end=4.8, chapter="先生（お手本）",
         captions=[(0, "先生：シミュレータだけが知る「瞳孔の本当の位置」を見て、数式（P制御）で操作する"),
                   (5, "実機では使えない情報を使うので必ず上手い。この操作を自動で記録してお手本にする")]),
    dict(kind="image", src="m2_demo_samples.png", seconds=5, chapter="お手本",
         captions=[(0, "お手本：カメラ画像と先生の指令（赤い矢印）の組を、3,000人分・約3.6万組 自動で集めた")]),
    dict(kind="clip", src="m3_student_demo.mp4", speed=0.5, chapter="生徒（模倣学習）",
         captions=[(0, "生徒：カメラ画像だけを見て、ジョイスティックの操作を直接出力するニューラルネット"),
                   (6.5, "お手本を真似て学習（行動クローニング、BC）→ 生徒が迷う場面を先生に聞き直して再学習（DAgger）"),
                   (13, "右下の小さな画像が、生徒が実際に見ている入力（96×96画素）")]),
    dict(kind="image", src="m3_compare.png", seconds=6, chapter="生徒の結果",
         captions=[(0, "学習に使っていない患者50人で成功率100%。DAgger で最終誤差 0.072 → 0.044 mm")]),
    dict(kind="clip", src="m4a_left_eye_modular_vs_e2e.mp4", speed=1.0, chapter="拡張A　左眼・近道学習",
         captions=[(0, "下：右眼だけで学んだ生徒は、左眼を見せても右眼の方へ移動してしまう（瞳孔ではなく顔の輪郭を手がかりにしていた＝近道学習）"),
                   (6.5, "上：瞳孔を検出して制御するモジュール型は、学習なしで左眼にも合う（成功率100%）")]),
    dict(kind="clip", src="m4g_delay_demo.mp4", speed=1.0, chapter="拡張G　画像の遅れ", layout="delay3",
         captions=[(0, "左から、画像の遅れ 0秒・0.2秒・0.3秒。遅れが大きいと行き過ぎて振動する（0.3秒で成功率0%）"),
                   (3.5, "理論（遅れのある制御）の予測どおり。動かす量を小さくすると成功率100%に戻る")]),
    dict(kind="clip", src="m4f_switch_demo.mp4", speed=1.0, chapter="拡張F　右眼⇄左眼の切替",
         captions=[(0, "「左眼へ」「右眼へ」という指示も入力する生徒。指示どおりに反対の眼へ移って合わせる"),
                   (5, "成功率100%（平均2.3秒）。指示を入力しない生徒は0%")]),
    dict(kind="clip", src="m4e_blink_demo.mp4", speed=0.5, chapter="拡張E　瞬き",
         captions=[(0, "患者が瞬きをする。上：瞬きを知らない生徒は、瞼が閉じている間に動いてしまう（平均0.5mm）"),
                   (3.5, "下：瞬きのお手本で学んだ生徒（直前の画像も見る）は、瞬きの間は待つ（平均0.05mm）")]),
    dict(kind="clip", src="m4h_focus_demo.mp4", speed=0.5, chapter="拡張H　ピント合わせ（前後）",
         captions=[(0, "前後のピント合わせ。焦点からずれると画像がぼける（右下）。上：手前から奥へ押して合わせる（成功率86%）"),
                   (4.6, "下：近づきすぎた位置から始めると、ぼけだけでは向きが分からず、さらに奥へ押してしまう")]),
    dict(kind="card", seconds=11, chapter="まとめ",
         lines=["まとめ",
                "・ロボットが細隙灯のジョイスティックを操作し、画像だけで瞳孔に合わせた（成功率100%）",
                "・模倣学習の弱点も見えた：近道学習（左眼）、画像の遅れ、手順の外の状況（近すぎ）",
                "・眼科医の手順（手前から奥へ押す）が、画像では決まらない向きを補っていることが分かった",
                "・今後：実際の眼の画像・実機での検証、層ごとのピント送り、安全装置"]),
]


def caption_band(chapter, text):
    """下の帯：章の名前（小さく灰色）と字幕（白）。長い字幕は2行に折り返す。"""
    band = Image.new("RGB", (W, H - AREA_H), BG)
    d = ImageDraw.Draw(band)
    if chapter:
        d.text((24, 8), chapter, font=font(20), fill=(150, 170, 200))
    f = font(28)
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > W - 48:
            lines.append(cur)
            cur = ""
        cur += ch
    lines.append(cur)
    for i in range(1, len(lines)):
        while lines[i] and lines[i][0] in "）)、。」":
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    lines = [line for line in lines if line]
    y = 40 if len(lines) == 1 else 34
    for line in lines[:2]:
        d.text((24, y), line, font=f, fill=(255, 255, 255))
        y += 38
    return np.array(band)


def fit(frame):
    """素材を 1280×600 の枠に、縦横比を保って中央に置く。"""
    img = Image.fromarray(frame)
    s = min(W / img.width, AREA_H / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    area = Image.new("RGB", (W, AREA_H), BG)
    area.paste(img, ((W - img.width) // 2, (AREA_H - img.height) // 2))
    return area


def delay3(frame):
    """M4-G の動画（3段）から細隙灯カメラ（各段の右半分）だけを切り出し、横に並べて見出しをつける。"""
    rh, half = frame.shape[0] // 3, frame.shape[1] // 2
    row = Image.fromarray(np.hstack([frame[i * rh:(i + 1) * rh, half:] for i in range(3)]))
    out = Image.new("RGB", (row.width, row.height + 60), BG)
    out.paste(row, (0, 60))
    d = ImageDraw.Draw(out)
    for i, label in enumerate(["遅れ 0秒", "遅れ 0.2秒", "遅れ 0.3秒"]):
        d.text((i * half + 20, 8), label, font=font(36), fill=(255, 255, 255))
    return np.array(out)


def speed_badge(area, speed):
    """右上に再生速度を表示（実時間との違いが分かるように）。"""
    label = "実時間" if speed == 1 else f"×{speed:g}（スロー）" if speed < 1 else f"×{speed:g}（早送り）"
    d = ImageDraw.Draw(area)
    f = font(22)
    w = d.textlength(label, font=f)
    d.rounded_rectangle((W - w - 36, 10, W - 12, 46), 8, fill=(0, 0, 0))
    d.text((W - w - 24, 13), label, font=f, fill=(255, 220, 90))


def title_overlay(area, lines):
    """最初の場面に、タイトルを半透明の帯で重ねる。"""
    over = Image.new("RGBA", area.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    d.rectangle((0, 380, W, 560), fill=(0, 0, 0, 170))
    d.text((40, 395), lines[0], font=font(56), fill=(255, 255, 255, 255))
    d.text((44, 480), lines[1], font=font(28), fill=(220, 230, 255, 255))
    return Image.alpha_composite(area.convert("RGBA"), over).convert("RGB")


def card(lines):
    area = Image.new("RGB", (W, AREA_H), BG)
    d = ImageDraw.Draw(area)
    d.text((60, 60), lines[0], font=font(48), fill=(255, 255, 255))
    for i, line in enumerate(lines[1:]):
        d.text((60, 160 + i * 80), line, font=font(27), fill=(230, 230, 230))
    return area


LAYOUTS = {"delay3": delay3}


def caption_at(captions, t):
    text = ""
    for t0, c in captions:
        if t >= t0:
            text = c
    return text


def segment_frames(seg):
    """1つの区切りを、30fps の 1280×720 の画像の列にする。"""
    caps = seg.get("captions", [])
    bands = {}
    if seg["kind"] == "clip":
        reader = imageio.get_reader(OUT / seg["src"])
        src_fps = reader.get_meta_data()["fps"]
        src = [f for f in reader]
        reader.close()
        if "end" in seg:
            src = src[:round(seg["end"] * src_fps)]
        n = int(len(src) / src_fps / seg["speed"] * FPS)
        get = lambda i: src[min(int(i / FPS * seg["speed"] * src_fps), len(src) - 1)]
    elif seg["kind"] == "image":
        still = imageio.imread(OUT / seg["src"])[..., :3]
        n = int(seg["seconds"] * FPS)
        get = lambda i: still
    else:
        n = int(seg["seconds"] * FPS)
        get = None
    still_area = card(seg["lines"]) if seg["kind"] == "card" else fit(get(0)) if seg["kind"] == "image" else None
    for i in range(n):
        area = still_area.copy() if still_area is not None else fit(LAYOUTS[seg["layout"]](get(i)) if "layout" in seg else get(i))
        if seg["kind"] == "clip":
            speed_badge(area, seg["speed"])
        if "title" in seg and i < 6 * FPS:     # タイトルは最初の6秒だけ
            area = title_overlay(area, seg["title"])
        text = caption_at(caps, i / FPS)
        if text not in bands:
            bands[text] = caption_band(seg["chapter"], text)
        yield np.vstack([np.array(area), bands[text]])


def main():
    path = OUT / "m5_submission.mp4"
    total = 0
    with imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1) as w:
        for seg in TIMELINE:
            n0 = total
            for frame in segment_frames(seg):
                w.append_data(frame)
                total += 1
            print(f"{total / FPS:6.1f} s  {seg.get('src', seg['chapter'])}  ({(total - n0) / FPS:.1f} s)")
    print("saved", path.relative_to(ROOT), f"{total / FPS:.1f} s")


if __name__ == "__main__":
    main()
