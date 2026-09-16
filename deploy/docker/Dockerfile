# syntax=docker/dockerfile:1
# ============================================================================================
# Immagine del controller h1_2_ctrl_squat per il PC di bordo del robot.
#
# Multi-stage: lo stadio `builder` compila SDK e controller (tutti i -dev, ~1.5 GB), lo stadio
# finale contiene SOLO cio' che serve a girare: binario, 4 librerie condivise, config, policy.
# Risultato ~120 MB. Stessa Dockerfile per x86_64 e aarch64: l'architettura la decide
# `docker build --platform` (o la macchina su cui costruisci), l'SDK Unitree e' prebuilt per
# entrambe e il CMakeLists del controller sceglie il tarball onnxruntime giusto.
#
# Contesto di build = la radice `Squat_Unitree_H1_2/` (vedi build.sh), con `unitree_sdk2/`
# copiato dentro `deploy/docker/_ctx/` dallo script. La policy NON e' dentro l'immagine: si monta
# a runtime (run.sh), cosi' cambi policy senza ricostruire.
# ============================================================================================

FROM ubuntu:20.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake g++ make ca-certificates \
        libyaml-cpp-dev libeigen3-dev libboost-program-options-dev libfmt-dev libspdlog-dev \
    && rm -rf /var/lib/apt/lists/*

# --- unitree_sdk2 (il nostro fork, con dds_wrapper e CycloneDDS prebuilt) --------------------
COPY deploy/docker/_ctx/unitree_sdk2 /src/unitree_sdk2
RUN cd /src/unitree_sdk2 && mkdir -p build && cd build \
    && cmake .. -DBUILD_EXAMPLES=OFF && make install

# --- controller --------------------------------------------------------------------------------
COPY deploy /src/Squat_Unitree_H1_2/deploy
RUN cd /src/Squat_Unitree_H1_2/deploy/robots/h1_2_squat && rm -rf build && mkdir build && cd build \
    && cmake .. && make -j"$(nproc)"

# ============================================================================================
FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
# libfmt e' l'unica libreria di sistema linkata dinamicamente (spdlog e' header-only)
RUN apt-get update && apt-get install -y --no-install-recommends libfmt-dev tmux \
    && rm -rf /var/lib/apt/lists/*

# CycloneDDS (installato da make install dell'SDK) e onnxruntime (vendorizzato)
COPY --from=builder /usr/local/lib/libddsc.so* /usr/local/lib/libddscxx.so* /usr/local/lib/
COPY --from=builder /src/Squat_Unitree_H1_2/deploy/thirdparty/*/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

# Layout identico a quello del PC di sviluppo: il controller risolve la policy come
# ../../../logs/rsl_rl/unitree_h1_2_squat relativo a robots/h1_2_squat/, e i log vanno in
# $HOME/unitree_ws/sim_logs. Entrambi sono punti di mount in run.sh.
ENV HOME=/root
WORKDIR /root/unitree_ws/Squat_Unitree_H1_2/deploy/robots/h1_2_squat
COPY --from=builder /src/Squat_Unitree_H1_2/deploy/robots/h1_2_squat/build/h1_2_ctrl_squat build/h1_2_ctrl_squat
COPY deploy/robots/h1_2_squat/config config
RUN mkdir -p /root/unitree_ws/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat /root/unitree_ws/sim_logs \
    && ./build/h1_2_ctrl_squat --help >/dev/null   # smoke test: il binario parte e trova le librerie

ENTRYPOINT ["./build/h1_2_ctrl_squat"]
CMD ["--help"]
