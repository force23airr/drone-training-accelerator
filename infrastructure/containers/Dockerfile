# Autonomous Flight Training Platform
# Multi-stage Dockerfile for training and deployment

# =============================================================================
# BASE STAGE - Common dependencies
# =============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash drone
WORKDIR /home/drone/app

# =============================================================================
# TRAINING STAGE - Full training environment
# =============================================================================
FROM base as training

# Install PyTorch with CUDA support (adjust version as needed)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY --chown=drone:drone . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER drone

# Default command
CMD ["python", "quickstart_train.py", "--help"]

# =============================================================================
# CPU TRAINING STAGE - Training without GPU
# =============================================================================
FROM base as training-cpu

# Install PyTorch CPU version
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY --chown=drone:drone . .

# Install package
RUN pip install -e .

USER drone

CMD ["python", "quickstart_train.py", "--help"]

# =============================================================================
# INFERENCE STAGE - Lightweight deployment image
# =============================================================================
FROM python:3.10-slim as inference

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash drone
WORKDIR /home/drone/app

# Install minimal dependencies for inference
RUN pip install --no-cache-dir \
    numpy \
    torch --index-url https://download.pytorch.org/whl/cpu \
    stable-baselines3

# Copy only necessary files
COPY --chown=drone:drone simulation/ ./simulation/
COPY --chown=drone:drone deployment/ ./deployment/
COPY --chown=drone:drone __init__.py .
COPY --chown=drone:drone setup.py .

RUN pip install -e .

USER drone

# =============================================================================
# ROS2 DEPLOYMENT STAGE - For hardware deployment
# =============================================================================
FROM ros:humble-ros-base as ros2-deployment

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    torch --index-url https://download.pytorch.org/whl/cpu \
    stable-baselines3

WORKDIR /ros2_ws/src

# Copy deployment package
COPY deployment/ros2-bridge/ ./drone_policy/

WORKDIR /ros2_ws

# Build ROS2 package
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Source setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
