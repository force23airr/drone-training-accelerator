# Installation Guide

Complete installation instructions for the Autonomous Flight Training Platform.

---

## System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| OS | Windows 10+, Ubuntu 20.04+, macOS 11+ |
| Python | 3.8, 3.9, 3.10, or 3.11 |
| RAM | 8 GB |
| Storage | 5 GB free space |
| CPU | 4+ cores recommended |

### Recommended (for training)
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with CUDA 11.8+ |
| VRAM | 8 GB+ |
| RAM | 16 GB+ |
| CPU | 8+ cores |

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-org/drone-training-accelerator.git
cd drone-training-accelerator

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Install package in development mode
pip install -e .

# 7. Verify installation
python -c "from simulation import BaseDroneEnv; print('Success!')"
```

### Method 2: Conda Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/drone-training-accelerator.git
cd drone-training-accelerator

# 2. Create conda environment
conda create -n drone-env python=3.10 -y
conda activate drone-env

# 3. Install PyTorch with CUDA (if GPU available)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
# For CPU only:
# conda install pytorch torchvision cpuonly -c pytorch -y

# 4. Install remaining dependencies
pip install -r requirements.txt
pip install -e .

# 5. Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Method 3: Docker Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/drone-training-accelerator.git
cd drone-training-accelerator

# 2. Build and run (GPU)
docker-compose up training-gpu

# Or for CPU-only
docker-compose up training-cpu

# 3. Access container
docker exec -it drone-training-gpu bash
```

---

## Detailed Dependency Installation

### Core Dependencies

```bash
# Physics simulation
pip install pybullet>=3.2.5

# Reinforcement Learning
pip install gymnasium>=0.28.0
pip install stable-baselines3>=2.0.0

# PyTorch (GPU)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch (CPU only)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### Optional Dependencies

```bash
# Development tools
pip install -e ".[dev]"
# Includes: pytest, black, flake8, mypy

# ROS2 integration
pip install -e ".[ros2]"
# Requires ROS2 to be installed separately

# Distributed training
pip install ray[rllib]>=2.5.0
```

---

## Platform-Specific Instructions

### Windows

**Prerequisites:**
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (if using GPU)

**Known Issues:**
- PyBullet GUI may require additional Visual C++ Redistributable
- Use PowerShell for best compatibility

```powershell
# Windows-specific installation
pip install pybullet --no-cache-dir
```

### Linux (Ubuntu/Debian)

**Prerequisites:**
```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip \
    python3-venv

# NVIDIA drivers (if using GPU)
sudo apt-get install -y nvidia-driver-525
```

**For headless servers:**
```bash
# Virtual display for PyBullet
sudo apt-get install -y xvfb
xvfb-run -a python your_script.py
```

### macOS

**Prerequisites:**
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10
```

**Note:** GPU acceleration via CUDA is not available on macOS. Use MPS (Apple Silicon) or CPU:
```python
# For Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## GPU Setup

### NVIDIA GPU (CUDA)

1. **Check GPU compatibility:**
```bash
nvidia-smi
```

2. **Install CUDA Toolkit:**
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Recommended: CUDA 11.8 or 12.1

3. **Install cuDNN:**
- Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- Extract and copy to CUDA directory

4. **Verify PyTorch CUDA:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### AMD GPU (ROCm) - Experimental

```bash
# Install ROCm PyTorch
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## Verification

### Quick Test
```bash
# Test imports
python -c "
from simulation import BaseDroneEnv, list_platforms
from training import ParallelTrainer, MissionSuite
print('All imports successful!')
print(f'Available platforms: {list_platforms()}')
"
```

### Full Test Suite
```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_environments.py -v
pytest tests/test_training.py -v
```

### GPU Test
```python
# test_gpu.py
import torch
from stable_baselines3 import PPO
from simulation import BaseDroneEnv, get_platform_config

print(f"CUDA available: {torch.cuda.is_available()}")

env = BaseDroneEnv(platform_config=get_platform_config("quadcopter_basic"))
model = PPO("MlpPolicy", env, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Model device: {model.device}")

# Quick training test
model.learn(total_timesteps=1000)
print("GPU training test passed!")
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'simulation'`
```bash
# Solution: Install package in development mode
pip install -e .
```

**Problem:** `ImportError: cannot import name 'spaces' from 'gymnasium'`
```bash
# Solution: Upgrade gymnasium
pip install --upgrade gymnasium
```

### PyBullet Issues

**Problem:** `PyBullet GUI won't open`
```bash
# Linux: Install display dependencies
sudo apt-get install libgl1-mesa-glx

# Or use headless mode
env = BaseDroneEnv(..., render_mode=None)
```

**Problem:** `pybullet build failed`
```bash
# Windows: Install Visual Studio Build Tools
# Linux: Install build-essential
sudo apt-get install build-essential

# Retry installation
pip install pybullet --no-cache-dir
```

### CUDA Issues

**Problem:** `CUDA out of memory`
```python
# Reduce batch size
trainer = ParallelTrainer(..., num_envs=2)

# Or use CPU
trainer = ParallelTrainer(..., device="cpu")

# Clear cache
import torch
torch.cuda.empty_cache()
```

**Problem:** `CUDA version mismatch`
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Reinstall PyTorch with matching CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Performance Issues

**Problem:** Training is slow
```python
# Use more parallel environments
trainer = ParallelTrainer(..., num_envs=8)

# Disable rendering during training
env = BaseDroneEnv(..., render_mode=None)

# Use SubprocVecEnv instead of DummyVecEnv (automatic for num_envs > 1)
```

---

## Updating

### Update to Latest Version
```bash
cd drone-training-accelerator
git pull origin main
pip install -e . --upgrade
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/  # Linux/macOS
rmdir /s /q venv  # Windows

# Or remove conda environment
conda deactivate
conda env remove -n drone-env
```

---

## Next Steps

After successful installation:
1. Read [GETTING_STARTED.md](GETTING_STARTED.md) for your first training session
2. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. Explore example scripts in `examples/`

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Search [GitHub Issues](https://github.com/your-org/drone-training-accelerator/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce
