ARG PYTORCH="2.7.0"
ARG CUDA="11.8"
ARG CUDNN="9"

ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4" \
        opencv-python \
        openvino \
        timm

WORKDIR /opt

RUN git clone https://github.com/isl-org/MiDaS.git midas && \
    cd midas && \
    git reset --hard 1645b7e1675301fdfac03640738fe5a6531e17d6

COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/midas/
COPY midas_* /usr/bin/

ENV PYTHONPATH=/opt/midas
