# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with source build for CPU

FROM ubuntu:18.04
LABEL maintainer="chasun@microsoft.com"
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:microsoft' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace
ENV DEBIAN_FRONTEND=noninteractive

#ADD . /code

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    vim \
    openssh-server \
    ca-certificates \
    g++ \
    gdb \
    gcc \
    make \
    net-tools

# Install python3
RUN apt-get install -y --no-install-recommends \
    python3-setuptools \
    python3-wheel \
    python3-pip \
    python3-numpy \
    aria2 && aria2c -q -d /tmp -o cmake-3.24.3-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.24.3/cmake-3.24.3-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.24.3-linux-x86_64.tar.gz --strip=1 -C /usr

RUN /etc/init.d/ssh restart
# ADD . /code
# # Prepare onnxruntime repository & build onnxruntime
# RUN cd /code && /bin/bash ./build.sh --skip_submodule_sync --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) 

# # Set environment and working directory
# ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
# ENV TRT_OSSPATH /workspace/TensorRT
# ENV TRT_MODELDIR /workspace/model
# ENV TRT_DATADIR /workspace/data
# ENV PATH="${PATH}:/usr/local/bin/ngc-cli"
#ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}
ENV _OSSPATH /workspace/onnxruntime
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:${OSS_PATH}/build/Linux:${LD_LIBRARY_PATHTRT_OSSPATH /workspace/TensorRTWORKDIR /workspace

# Build
# C++
# RUN cd /code && /bin/bash ./build.sh --config RelWithDebInfo --build_shared_lib --parallel

USER trtuser
RUN ["/bin/bash"]