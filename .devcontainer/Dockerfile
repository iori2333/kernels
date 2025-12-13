FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
  sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y \
  python3 python3-pip python-is-python3 git && \
  rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

RUN pip install --break-system-packages uv

# install some essential packages
RUN apt-get update && apt-get install -y \
  build-essential clang clangd cmake \
  wget curl nvtop htop vim && \
  rm -rf /var/lib/apt/lists/*
