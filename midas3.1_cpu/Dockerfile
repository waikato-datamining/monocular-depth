ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.7.0 torchvision \
    --index-url  https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4" \
        opencv-python \
        openvino \
        imutils \
        timm==0.6.13

WORKDIR /opt

RUN git clone https://github.com/isl-org/MiDaS.git midas && \
    cd midas && \
    git reset --hard 1645b7e1675301fdfac03640738fe5a6531e17d6

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY bash.bashrc /etc/bash.bashrc
COPY predict*.py /opt/midas/
COPY midas_* /usr/bin/

ENV PYTHONPATH=/opt/midas
