#!/bin/bash
# =============================================================================
# Docker Entrypoint Script
# =============================================================================
# Handles various run modes and environment setup
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Start virtual display for headless rendering
# -----------------------------------------------------------------------------
start_xvfb() {
    if [ -z "$DISPLAY" ] || [ "$DISPLAY" = ":99" ]; then
        echo "Starting virtual display (Xvfb)..."
        Xvfb :99 -screen 0 1920x1080x24 &
        export DISPLAY=:99
        sleep 1
    fi
}

# -----------------------------------------------------------------------------
# Start PX4 SITL if requested
# -----------------------------------------------------------------------------
start_px4_sitl() {
    if [ -d "/opt/px4" ]; then
        echo "Starting PX4 SITL..."
        cd /opt/px4
        HEADLESS=1 make px4_sitl none_iris &
        cd /app
        sleep 5  # Wait for PX4 to start
    else
        echo "Warning: PX4 not installed in this image"
    fi
}

# -----------------------------------------------------------------------------
# Main entrypoint logic
# -----------------------------------------------------------------------------
main() {
    # Start Xvfb for headless rendering
    start_xvfb

    # Handle special commands
    case "${1:-}" in
        --api)
            echo "Starting API server..."
            shift
            exec uvicorn api.main:app --host 0.0.0.0 --port 8080 "$@"
            ;;

        --train)
            echo "Starting training..."
            shift
            exec python -m training.run "$@"
            ;;

        --px4)
            echo "Starting with PX4 SITL..."
            start_px4_sitl
            shift
            exec "$@"
            ;;

        --jupyter)
            echo "Starting Jupyter notebook..."
            exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
            ;;

        --tensorboard)
            echo "Starting TensorBoard..."
            exec tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006
            ;;

        --test)
            echo "Running tests..."
            exec pytest tests/ -v "$@"
            ;;

        --shell|bash|sh)
            exec /bin/bash
            ;;

        --help|-h)
            cat << EOF
Drone Training Accelerator Docker Image

Usage: docker run [docker-options] drone-accelerator [command] [args]

Commands:
    --api           Start the REST API server (port 8080)
    --train         Run training with specified config
    --px4           Start with PX4 SITL integration
    --jupyter       Start Jupyter notebook server (port 8888)
    --tensorboard   Start TensorBoard (port 6006)
    --test          Run test suite
    --shell, bash   Start interactive shell
    --help          Show this help message

Examples:
    # Run a quick test
    docker run drone-accelerator python test_x47b.py

    # Start API server
    docker run -p 8080:8080 drone-accelerator --api

    # Train with GPU
    docker run --gpus all drone-accelerator --train --config configs/ppo.yaml

    # Interactive development
    docker run -it -v \$(pwd):/app drone-accelerator bash

    # Run with PX4 SITL
    docker run --network host drone-accelerator:px4-sitl --px4 python test_px4_sitl.py

Environment Variables:
    CUDA_VISIBLE_DEVICES    GPU selection for training
    WANDB_API_KEY           Weights & Biases logging
    DISPLAY                 X display for rendering (default: :99)

EOF
            exit 0
            ;;

        "")
            # No arguments - show platform info
            echo "Drone Training Accelerator"
            echo "========================="
            python -c "
from simulation import list_platforms, PX4_SITL_AVAILABLE
print('Available platforms:', list_platforms())
print('PX4 SITL available:', PX4_SITL_AVAILABLE)
print()
print('Run with --help for usage information')
"
            ;;

        *)
            # Pass through to exec
            exec "$@"
            ;;
    esac
}

main "$@"
