FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# ENV APP_USER=edyalenquer
# ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update  \
    && apt-get upgrade -y \
    && apt-get install -y \
        sudo \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libcairo2-dev \
        pkg-config \ 
        libglib2.0-0 \
        curl \
        wget \
        bash \
    && rm -rf /var/lib/apt/lists

    
# # Create a non-root user
# RUN useradd -ms /bin/bash $APP_USER

# # Give the non-root user sudo privileges without password
# RUN echo '$APP_USER ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# # Set the user for the container and the working directory
# WORKDIR /home/$APP_USER
# RUN chown -R $APP_USER:$APP_USER /home/$APP_USER
# USER $APP_USER

# Upgrade pip

RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install any python packages you need
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

# # Set the entrypoint
ENTRYPOINT [ "python3" ]