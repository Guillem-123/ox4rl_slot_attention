# Use NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    git \
    # Add OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Set working directory
WORKDIR /app

# Clone your repository (replace with your actual repo URL)
# If you're copying local files instead, use COPY instead of git clone
COPY . /app/

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install -r /app/requirements_colab.txt
RUN python -m pip install torchmetrics
# Install the package in development mode
RUN python -m pip install -e .

# Create necessary directories
RUN mkdir -p /app/data
RUN mkdir -p /app/runs

# Set environment variables for NVIDIA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
# Compute root images\n\
python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f train -m SPACE -g Pong --compute_root_images\n\
python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f train -m SPACE -g Pong\n
python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f validation -m SPACE -g Pong\n
python -m ox4rl.dataset_creation.create_dataset_using_OCAtari -f test -m SPACE -g Pong\n
python -m execution_scripts.create_motion_for_slot --config_file ./configs/slot_atari_pong.yaml --dataset_mode train\n
python -m execution_scripts.create_motion_for_slot --config_file ./configs/slot_atari_pong.yaml --dataset_mode validation\n
python -m execution_scripts.create_motion_for_slot --config_file ./configs/slot_atari_pong.yaml --dataset_mode test\n
\n\
# Train the model\n\
python -m ox4rl.execution_scripts.trainSlotAtari --config_file ./ox4rl/configs/slot_atari_pong.yaml\n\
' > /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Copy the create_latents script
COPY create_latents.py /app/

# Install PyYAML if not already installed
RUN pip install pyyaml

# Run the script to generate placeholder latent files
# RUN python /app/create_latents.py

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 