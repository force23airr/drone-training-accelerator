# =============================================================================
# Drone Training Accelerator - Docker Image
# =============================================================================
# Multi-stage build for optimized image size
#
# Usage:
#   Build:    docker build -t drone-accelerator .
#   Run:      docker run -it drone-accelerator
#   With GPU: docker run --gpus all -it drone-accelerator
#   API mode: docker run -p 8080:8080 drone-accelerator --api
#
# For development:
#   docker run -it -v $(pwd):/app drone-accelerator bash
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# -----------------------------------------------------------------------------
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    # Build tools
    build-essential \
    cmake \
    git \
    # Graphics (for PyBullet headless rendering)
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    xvfb \
    # Networking (for PX4 SITL)
    netcat-openbsd \
    # Utilities
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Stage 2: Python dependencies
# -----------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML/RL dependencies
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    stable-baselines3>=2.0.0 \
    gymnasium>=0.29.0 \
    pybullet>=3.2.5 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    tensorboard>=2.13.0 \
    wandb>=0.15.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0

# -----------------------------------------------------------------------------
# Stage 3: Final image with application code
# -----------------------------------------------------------------------------
FROM dependencies AS final

WORKDIR /app

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/models /app/logs /app/data

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV DISPLAY=:99
ENV PYBULLET_EGL=1

# Expose ports
# 8080: REST API
# 14560: PX4 SITL MAVLink
# 6006: TensorBoard
EXPOSE 8080 14560 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from simulation import BaseDroneEnv; print('OK')" || exit 1

# Default entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "-c", "from simulation import list_platforms; print('Available platforms:', list_platforms())"]


# -----------------------------------------------------------------------------
# Stage: Development image with additional tools
# -----------------------------------------------------------------------------
FROM final AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    mypy>=1.0.0 \
    flake8>=6.0.0 \
    jupyter>=1.0.0 \
    ipython>=8.0.0

# Install vim and other dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash"]


# -----------------------------------------------------------------------------
# Stage: PX4 SITL image with flight controller
# -----------------------------------------------------------------------------
FROM final AS px4-sitl

# Install PX4 dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre \
    ant \
    && rm -rf /var/lib/apt/lists/*

# Clone and build PX4 (lightweight - just SITL)
RUN git clone --depth 1 --branch v1.14.0 https://github.com/PX4/PX4-Autopilot.git /opt/px4 \
    && cd /opt/px4 \
    && git submodule update --init --recursive --depth 1 \
    && pip install --no-cache-dir -r /opt/px4/Tools/setup/requirements.txt

# Build PX4 SITL
WORKDIR /opt/px4
RUN DONT_RUN=1 make px4_sitl none_iris

# Back to app directory
WORKDIR /app

ENV PX4_HOME=/opt/px4

CMD ["python", "test_px4_sitl.py"]


# -----------------------------------------------------------------------------
# Stage: API server image
# -----------------------------------------------------------------------------
FROM final AS api-server

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    pydantic>=2.0.0 \
    python-multipart>=0.0.6

# Copy API code
COPY api/ /app/api/

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]


# -----------------------------------------------------------------------------
# Stage: Training image optimized for GPU clusters
# -----------------------------------------------------------------------------
FROM final AS training

# Additional ML libraries for training
RUN pip install --no-cache-dir \
    ray[rllib]>=2.5.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    optuna>=3.2.0

# Set up for distributed training
ENV RAY_DISABLE_MEMORY_MONITOR=1

CMD ["python", "-m", "training.run", "--help"]
