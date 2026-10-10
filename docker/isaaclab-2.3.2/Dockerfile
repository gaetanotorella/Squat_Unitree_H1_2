# Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + rsl-rl-lib 5.0.1: lo stack con cui sono stati addestrati TUTTI i
# run della tesi (R4-R8, policy finale 2026-09-13_21-27-39; codice al tag isaaclab-2.3.2).
#
# Perche' esiste: dal 2026-10 sul server del lab Isaac e' disponibile solo come Isaac Lab 3.0.0-rc1 /
# Isaac Sim 6.1.0-rc, e il porting del repo a 3.0 ha cambiamenti silenziosi (kp/kd = 0 nel deploy.yaml,
# forze di contatto solo normali, ginocchia a cilindro, nuovo percorso degli attuatori espliciti) e
# renderebbe i run nuovi non confrontabili con quelli vecchi. Concordato con gli admin: immagine
# personale, da cancellare a fine tesi (vedi README.md, sezione "Pulizia").
#
# Ricalca docker/Dockerfile.base di Isaac Lab v2.3.2, con tre differenze:
#   - Isaac Lab clonato dal tag invece che copiato dal contesto (il contesto di build resta questa cartella);
#   - learning framework solo rsl_rl, poi rsl-rl-lib portato a 5.0.1 come sul PC di training originale;
#   - il container gira come utente 1234 (quello previsto da Isaac Sim 5.1), non root, cosi' i file che
#     scrive nelle cartelle montate non sono di root e restano cancellabili senza sudo.
ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

ARG ISAACLAB_VERSION=v2.3.2
ARG RSL_RL_VERSION=5.0.1

ENV ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="isaaclab-2.3.2 (tesi H1_2 squat)" \
      org.opencontainers.image.description="Isaac Sim ${ISAACSIM_VERSION} + Isaac Lab ${ISAACLAB_VERSION} + rsl-rl-lib ${RSL_RL_VERSION}"

SHELL ["/bin/bash", "-c"]
USER root

# dipendenze di sistema di Isaac Lab (come Dockerfile.base) + ffmpeg per i video di media.py
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libglib2.0-0 ncurses-term wget ffmpeg && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${ISAACLAB_VERSION} https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH} && \
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim && \
    chmod +x ${ISAACLAB_PATH}/isaaclab.sh

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir toml && \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# come Dockerfile.base: segnaposto che il NVIDIA container toolkit sovrascrive al lancio
RUN touch /bin/nvidia-smi /bin/nvidia-debugdump /bin/nvidia-persistenced \
          /bin/nvidia-cuda-mps-control /bin/nvidia-cuda-mps-server /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && touch /var/run/nvidia-persistenced/socket

# estensioni Isaac Lab + rsl_rl (porta con se' rsl-rl-lib 3.1.2 e onnxscript), poi la versione usata nei run
RUN PIP_NO_CACHE_DIR=1 ${ISAACLAB_PATH}/isaaclab.sh --install rsl_rl && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir "rsl-rl-lib==${RSL_RL_VERSION}" psutil argcomplete && \
    (${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog || true)

# Il nostro pacchetto NON e' nell'immagine: il repo e' montato a runtime in /workspace/Squat_Unitree_H1_2 e
# un .pth lo mette nel path (equivale a `pip install -e`, ma senza scrivere nulla a runtime nel site-packages,
# che non e' dell'utente 1234). Cosi' una modifica al codice non richiede di ricostruire l'immagine.
RUN SITE=$(${ISAACSIM_ROOT_PATH}/python.sh -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    echo "/workspace/Squat_Unitree_H1_2/source/unitree_rl_lab" > "${SITE}/unitree_rl_lab_tesi.pth"

# Il repo montato appartiene all'utente dell'host, non a 1234: senza questo git (che rsl_rl usa per salvare
# lo stato del codice nel run) si rifiuta di leggerlo ("dubious ownership").
RUN git config --system --add safe.directory '*'

# punti di mount delle cache (da docs Isaac Sim 5.1, container rootless) e del repo
RUN mkdir -p /isaac-sim/.cache /isaac-sim/.nv/ComputeCache /isaac-sim/.nvidia-omniverse/logs \
             /isaac-sim/.nvidia-omniverse/config /isaac-sim/.local/share/ov/data /isaac-sim/.local/share/ov/pkg \
             /workspace/Squat_Unitree_H1_2 /workspace/unitree_ros && \
    chown -R 1234:1234 /isaac-sim/.cache /isaac-sim/.nv /isaac-sim/.nvidia-omniverse /isaac-sim/.local

USER 1234:1234
WORKDIR /workspace/Squat_Unitree_H1_2
ENV UNITREE_ROS_DIR=/workspace/unitree_ros
ENTRYPOINT []
CMD ["bash"]
