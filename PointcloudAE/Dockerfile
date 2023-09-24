# Base image
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Setup environment
ENV TERM xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.6/dist-packages/torch/lib/
ENV PYTHONPATH=${PYTHONPATH}:/pcr

# Install system packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential cmake git \
    libjpeg8-dev zlib1g-dev \
    pybind11-dev python3 python3-dev python3-pip \
    vim screen wget curl tree zip unzip bc \
    && rm -rf /var/lib/apt/lists/*

# Copy the libary to the docker image
COPY ./ pcr/

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools==59.6.0
RUN pip3 install -r pcr/requirements.txt

# Install depoco and 3rdparty dependencies
RUN cd pcr/metrics_from_point_flow/pytorch_structural_losses/ && make clean && make

WORKDIR /pcr
