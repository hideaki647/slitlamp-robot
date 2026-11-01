# 細隙灯アライメントロボット（Slit-lamp alignment robot）

ロボットアームが細隙灯顕微鏡の**ジョイスティックを握って操作**し、**カメラ画像だけ**で患者の瞳孔に合わせる。物理シミュレータ MuJoCo 上で、模倣学習（行動クローニング＋DAgger）により学習した。

A robot arm grasps the joystick of a slit-lamp microscope and aligns it to the patient's pupil from camera images only, trained by imitation learning (BC + DAgger) in MuJoCo.

- 動画：https://youtu.be/y5pibef0mDA
- レポート：[report/report.pdf](report/report.pdf)
- **患者はすべて合成CG**です。実在の患者の画像・情報、人間の医師の操作データは使っていません。医療機器ではありません。

## 概要
```
【先生】シミュレータだけが知る「瞳孔の本当の位置」を見て、P制御で操作（特権情報）
   ↓ 操作を自動で記録（合成患者3,000人、3.6万組）
【お手本】細隙灯カメラの画像（96×96）＋ 先生の指令 (Δx, Δz)
   ↓ 行動クローニング（BC）→ DAgger
【生徒】画像だけを見て指令を出す CNN（空間ソフトマックス）
   ↓ 指令 → ジョイスティックの角度 → 逆運動学 → 6関節アーム → 本体が動く
```
- 人の手のための操作インタフェースを、ロボットがそのまま握る＝既存の細隙灯を改造せずに遠隔化できる構成。
- ジョイスティックの傾き → 本体の左右・前後、ハンドルの回転 → 高さ（MuJoCo の等式拘束で再現）。

## 主な結果（学習に使っていない合成患者50人）
| 方式 | 成功率 | 収束時間 | 最終誤差 |
|---|---|---|---|
| 先生（特権情報、P制御） | 100% | 0.43秒 | 0.008mm |
| 生徒 BC（画像のみ） | 100% | 0.44秒 | 0.058mm |
| 生徒 BC＋DAgger（画像のみ） | 100% | 0.44秒 | 0.027mm |

成功＝光軸と瞳孔中心のずれ1mm未満を0.5秒保つこと。

### 拡張実験
| 実験 | 分かったこと |
|---|---|
| 左眼・モジュール型との比較 | 右眼だけで学んだ生徒は左眼を無視して右眼へ移動（近道学習）。瞳孔検出＋P制御のモジュール型は学習なしで左眼も100% |
| 画像の遅れ | 0.2秒で約8%、0.3秒で0%（制御理論の予測どおり）。遅れに合わせてゲインを下げると学習し直さずに100% |
| 右眼⇄左眼の切替 | 「右眼へ／左眼へ」の指示を入力する生徒で100%（指示なしは0%） |
| 瞬き | 瞬きを知らない生徒は瞼が閉じている間に動く（平均0.5mm）。瞬きのお手本で学ぶと「待つ」（0.05mm） |
| 前後のピント合わせ | 手前から奥へ押す臨床の手順なら86%。近づきすぎから始めると、ぼけだけでは向きが分からず奥へ押し込む |

詳しくはレポートと [docs/](docs/)（各段階の解説、日本語）を参照。

## 動かし方
Docker を使う（Windows では MuJoCo が直接動かない環境があったため）。GPU は不要（CPU のみ）。

```bash
docker compose build                     # 初回（Python 3.12・MuJoCo 3.15・PyTorch CPU 版）
docker compose run --rm sim python scripts/render_snapshot.py   # シーンの画像 → outputs/
```

### 再現の手順（`slitlamp_robot` フォルダで）
大きなデータ（`data/`）と学習済みの重み（`checkpoints/`）はリポジトリに含めていない。次の順に作り直せる（16並列のCPUで、合計1〜2時間程度）。

```bash
# M1：アームがジョイスティックを操作する動画
docker compose run --rm sim python scripts/m1_demo.py
# M2：先生の評価とお手本の収集（3,000人 × 1.2秒）
docker compose run --rm sim python scripts/eval_policy.py --policy expert
docker compose run --rm sim python learning/collect_demos.py --episodes 3000
# M3：生徒の学習（BC → DAgger 4ラウンド）と評価
docker compose run --rm sim python learning/train_bc.py
docker compose run --rm sim python learning/dagger.py --rounds 4
docker compose run --rm sim python scripts/eval_policy.py --policy student:checkpoints/dagger_r4.pt --tag dagger
docker compose run --rm sim python scripts/m3_student_demo.py --ckpt checkpoints/dagger_r4.pt
```
拡張実験（M4-A・G・F・E・H）のコマンドは、各解説（`docs/解説_M4*.md`）の末尾にある。提出用の動画は `scripts/m5_make_video.py` で、各段階の動画をつないで作る。

## フォルダ構成
| フォルダ | 中身 |
|---|---|
| `models/` | シーンの設計図（MJCF、`scene.xml`）、瞼の形 |
| `sim/` | 環境（`env.py`）、ジョイスティック ⇄ 本体の換算と把持（`slitlamp_api.py`）、逆運動学（`arm_ik.py`） |
| `expert/` | 先生（特権情報の P制御） |
| `learning/` | お手本の収集、生徒のモデル、BC・DAgger の学習 |
| `policies/` | モジュール型（瞳孔検出＋P制御）、両眼切替の比較方式 |
| `scripts/` | 評価、グラフ、動画 |
| `outputs/` | 評価結果（JSON）とグラフ。動画は容量のため含めない |
| `docs/` | 各段階の解説と計画（日本語） |
| `report/` | レポート（HTML 原稿と PDF） |

## 簡略化と限界
- 眼は球と円盤の簡略な CG（角膜・水晶体・睫毛・眼球運動なし）。焦点ぼけは深度画像に基づく後処理の近似。
- 把持は溶接拘束（握った状態から開始）、ジョイスティックの傾き角と本体の位置は一対一。
- お手本は人間の医師ではなく、特権情報を使う P制御の先生。

## ライセンス
[MIT License](LICENSE)

## 参考文献
- S. Ross, G. Gordon, J. A. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011.
- M. Laskey et al. DART: Noise Injection for Robust Imitation Learning. CoRL 2017.
- S. Levine, C. Finn, T. Darrell, P. Abbeel. End-to-End Training of Deep Visuomotor Policies. JMLR 2016.
- E. Todorov, T. Erez, Y. Tassa. MuJoCo: A physics engine for model-based control. IROS 2012.
