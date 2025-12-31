@echo off
REM install_gym_pybullet_drones.bat
REM Installation script for gym-pybullet-drones integration (Windows)

echo ==============================================
echo Installing gym-pybullet-drones and dependencies
echo ==============================================

echo.
echo [1/4] Installing gym-pybullet-drones...
pip install gym-pybullet-drones
if errorlevel 1 goto error

echo.
echo [2/4] Installing additional dependencies...
pip install pybullet>=3.2.5
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install pillow
if errorlevel 1 goto error

echo.
echo [3/4] Installing visualization dependencies...
pip install matplotlib>=3.5.0
if errorlevel 1 goto error

echo.
echo [4/4] Verifying installation...
python -c "import gym_pybullet_drones; print('✓ gym-pybullet-drones installed successfully')"
if errorlevel 1 goto error

python -c "from gym_pybullet_drones.envs.BaseAviary import BaseAviary; print('✓ BaseAviary import successful')"
if errorlevel 1 goto error

echo.
echo ==============================================
echo Installation complete!
echo ==============================================
echo.
echo You can now use gym-pybullet-drones environments.
echo.
goto end

:error
echo.
echo ERROR: Installation failed!
echo Please check the error messages above.
exit /b 1

:end
