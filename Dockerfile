# 細隙灯アライメントロボット用の実行環境
# ベース：Python 3.12 の軽量Linux（講義の roboops-workshop と同じ系統）
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # 画面なしで描画する方式。egl が動かなければ osmesa に切り替える
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

# 描画用ライブラリ（EGL / OSMesa）と動画書き出し用の ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libegl-mesa0 \
        libgl1 \
        libgles2 \
        libgl1-mesa-dri \
        libosmesa6 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch は CPU 版を専用サーバーから入れる（GPU版より大幅に小さい）
RUN pip install torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# プロジェクトはここにマウントされる（compose.yaml 参照）
WORKDIR /workspace
CMD ["bash"]
