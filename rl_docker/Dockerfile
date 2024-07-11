# for 3070
# FROM nvcr.io/nvidia/pytorch:21.09-py3 
# for 4090
FROM nvcr.io/nvidia/pytorch:22.12-py3 


# set WORKDIR
WORKDIR /home/root/rl_ws

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# replace mirror
RUN apt-get install -q -y --no-install-recommends wget \
&& wget http://archive.ubuntu.com/ubuntu/pool/main/c/ca-certificates/ca-certificates_20230311ubuntu0.20.04.1_all.deb \
&& dpkg -i ./ca-certificates_20230311ubuntu0.20.04.1_all.deb
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse\n" > /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse\n" >> /etc/apt/sources.list && \
  echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse\n" >> /etc/apt/sources.list

# dependencies for gym
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs \
 libsm6 \
 xauth \
 x11-apps

# change pip mirror
RUN python -m pip install --upgrade pip \
&& pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# pip install requirements
COPY requirement.txt .
RUN pip install -r requirement.txt

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
