FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update  \
    && apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        curl \
        wget \
        bash \
    && rm -rf /var/lib/apt/lists

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install any python packages you need
COPY requirements.txt /app

RUN python3 -m pip install -r requirements.txt


# # Set the entrypoint
ENTRYPOINT [ "python3" ]