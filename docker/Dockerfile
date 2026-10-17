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

# Web GUI.
RUN pip install --no-cache-dir "viser==1.1.1" "plotly==7.1.0"

# Run as the host user so files written to the bind-mounted repo are not root-owned.
RUN (getent passwd ${UID} | cut -d: -f1 | xargs -r userdel -r) 2>/dev/null; \
    groupadd -o -g ${GID} dev 2>/dev/null || true; \
    useradd -o -m -u ${UID} -g ${GID} -s /bin/bash dev
USER dev

ENV PYTHONPATH=/workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]
