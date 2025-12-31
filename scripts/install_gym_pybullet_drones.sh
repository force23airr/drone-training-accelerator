#!/bin/bash
# install_gym_pybullet_drones.sh
# Installation script for gym-pybullet-drones integration

set -e  # Exit on error

echo "=============================================="
echo "Installing gym-pybullet-drones and dependencies"
echo "=============================================="

# Install gym-pybullet-drones
echo ""
echo "[1/4] Installing gym-pybullet-drones..."
pip install gym-pybullet-drones

# Install additional dependencies if needed
echo ""
echo "[2/4] Installing additional dependencies..."
pip install pybullet>=3.2.5
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pillow

# Install optional visualization dependencies
echo ""
echo "[3/4] Installing visualization dependencies..."
pip install matplotlib>=3.5.0

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
python -c "import gym_pybullet_drones; print('✓ gym-pybullet-drones installed successfully')"
python -c "from gym_pybullet_drones.envs.BaseAviary import BaseAviary; print('✓ BaseAviary import successful')"
python -c "from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary; print('✓ CtrlAviary import successful')"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "You can now use gym-pybullet-drones environments:"
echo "  - BaseAviary: Base environment class"
echo "  - CtrlAviary: Control-based aviary"
echo "  - VelocityAviary: Velocity control"
echo ""
echo "Example usage:"
echo "  from gym_pybullet_drones.envs import CtrlAviary"
echo "  env = CtrlAviary()"
echo ""
