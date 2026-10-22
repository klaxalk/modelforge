# modelforge development image: COLMAP (C++) + pycolmap + pyceres built from the
# pinned submodules, sharing one Ceres and pybind11 3.1.0 so that pycolmap's
# ceres::Problem / CostFunction objects are usable through pyceres.
#
# CPU (default):  docker compose build
# CUDA:           docker compose -f compose.yaml -f compose.gpu.yaml build

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ARG CUDA_ENABLED=OFF
ARG CUDA_ARCHITECTURES="75;80;86;89;90"
# Parallel compile jobs; pycolmap binding TUs need ~2 GB each.
ARG JOBS=6
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive
# openimageio-tools and libopencv-core-dev work around Ubuntu 24.04 OpenImageIO
# packaging: its CMake targets reference /usr/bin/iconvert and /usr/include/opencv4.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-core-dev \
        build-essential cmake ninja-build git ccache ca-certificates curl \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libceres-dev libsuitesparse-dev libflann-dev libfreeimage-dev \
        libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev \
        libcgal-dev libopenimageio-dev openimageio-tools libcurl4-openssl-dev libssl-dev \
        python3-dev python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/ccache \
    CMAKE_BUILD_PARALLEL_LEVEL=${JOBS} \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/colmap/bin:${PATH}

RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "pybind11==3.1.0" "scikit-build-core>=0.10" build ninja \
        numpy opencv-python-headless pillow pytest

# COLMAP C++ library + CLI (no Qt GUI, no ONNX for now).
COPY resources/colmap /src/colmap
RUN --mount=type=cache,target=/ccache \
    cmake -S /src/colmap -B /build/colmap -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DCUDA_ENABLED=${CUDA_ENABLED} "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DGUI_ENABLED=OFF -DONNX_ENABLED=OFF -DIPO_ENABLED=OFF -DTESTS_ENABLED=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build /build/colmap -j ${JOBS} && cmake --install /build/colmap && \
    rm -rf /build/colmap

RUN --mount=type=cache,target=/ccache \
    CMAKE_PREFIX_PATH=/opt/colmap pip install --no-cache-dir --no-build-isolation \
        -C cmake.define.GENERATE_STUBS=OFF /src/colmap

COPY resources/pyceres /src/pyceres
RUN --mount=type=cache,target=/ccache \
    pip install --no-cache-dir --no-build-isolation /src/pyceres

# Fail the build early if the cross-module type sharing is broken.
RUN python -c "import pyceres, pycolmap; print('pycolmap', pycolmap.__version__, 'pyceres', pyceres.__version__)"

# Web GUI; mesh / splat PLY I/O.
RUN pip install --no-cache-dir "viser==1.1.1" "plotly==7.1.0" "trimesh==5.1.0" "plyfile==1.1.5"

# nerfstudio (splatfacto) in its own venv (CUDA image only, NERFSTUDIO=ON). It pins
# viser==1.0.0 and opencv 4.10 and wants numpy<2, which clash with /opt/venv (and needs
# pillow<12: its pil_to_numpy uses a Pillow-internal encoder API that changed in 12); the two
# environments only exchange files (transforms.json, PLY). Called as /opt/nerfstudio/bin/ns-*,
# never put on PATH. torch for CUDA 12.8 (Blackwell-capable) instead of nerfstudio's cu118;
# gsplat's CUDA kernels are compiled here for GSPLAT_ARCH_LIST, so training starts at once.
# gsplat comes from its git tag: the PyPI sdist lacks the glm submodule the kernels include.
# tiny-cuda-nn is not installed: only the NeRF methods need it, not splatfacto.
ARG NERFSTUDIO=OFF
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG GSPLAT_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"
COPY resources/nerfstudio /src/nerfstudio
RUN if [ "${NERFSTUDIO}" = "ON" ]; then \
      python3 -m venv /opt/nerfstudio && \
      /opt/nerfstudio/bin/pip install --no-cache-dir --upgrade pip setuptools wheel ninja && \
      /opt/nerfstudio/bin/pip install --no-cache-dir "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
          --index-url "${TORCH_INDEX_URL}" && \
      TORCH_CUDA_ARCH_LIST="${GSPLAT_ARCH_LIST}" MAX_JOBS=${JOBS} \
          /opt/nerfstudio/bin/pip install --no-cache-dir --no-build-isolation \
          "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" && \
      /opt/nerfstudio/bin/pip install --no-cache-dir "numpy<2" "pillow<12" /src/nerfstudio && \
      /opt/nerfstudio/bin/python -c "import torch, gsplat, nerfstudio; from gsplat.cuda._backend import _C; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gsplat', gsplat.__version__, 'prebuilt kernels', _C is not None)" ; \
    fi
ENV NERFSTUDIO_BIN=/opt/nerfstudio/bin

# ROS 2 bags (project kind "rosbag"): pure-Python reader for sqlite3 / MCAP bags, no ROS install needed.
RUN pip install --no-cache-dir "rosbags==0.11.8"

# C-RADIOv3 for semantic search (modelforge/semantic.py), CUDA image only: RADIO needs einops, timm >= 1.0
# and transformers, which clash with nerfstudio's pins (timm 0.6.7), so they go to their own folder that
# only the semantic subprocesses put first on PYTHONPATH; torch and numpy stay nerfstudio's (--no-deps).
# Model weights are downloaded on first use into build/ (MODELFORGE_MODEL_CACHE).
RUN if [ "${NERFSTUDIO}" = "ON" ]; then \
      /opt/nerfstudio/bin/python -m pip install --no-cache-dir --no-deps --target /opt/radio-pkgs \
        einops==0.8.2 timm==1.0.30 transformers==4.57.6 tokenizers==0.22.2 safetensors==0.8.0 \
        huggingface_hub==0.36.2 hf-xet==1.7.0 regex==2026.9.29 sentencepiece==0.2.2 plyfile==1.1.5; \
    fi

# Browser tab title of the GUI: viser's page hardcodes <title>Viser</title> and never changes it.
RUN sed -i 's|<title>Viser</title>|<title>ModelForge</title>|' \
        "$(python -c 'import os, viser; print(os.path.dirname(viser.__file__))')/client/build/index.html" && \
    grep -q "<title>ModelForge</title>" \
        "$(python -c 'import os, viser; print(os.path.dirname(viser.__file__))')/client/build/index.html"
# Camera keys R / F for up / down instead of viser's E / Q (the user asked).
COPY docker/patch_viser_keys.sh /tmp/patch_viser_keys.sh
RUN sh /tmp/patch_viser_keys.sh \
        "$(python -c 'import os, viser; print(os.path.dirname(viser.__file__))')/client/build/index.html"

# Run as the host user so files written to the bind-mounted repo are not root-owned.
RUN (getent passwd ${UID} | cut -d: -f1 | xargs -r userdel -r) 2>/dev/null; \
    groupadd -o -g ${GID} dev 2>/dev/null || true; \
    useradd -o -m -u ${UID} -g ${GID} -s /bin/bash dev
USER dev

ENV PYTHONPATH=/workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]
