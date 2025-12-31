#!/usr/bin/env python3
"""
Example: Export Trained Model to ONNX

Export a trained policy for deployment on embedded systems or
other platforms that support ONNX.
"""

import argparse
from pathlib import Path

from deployment import PolicyExporter, OnnxInferenceEngine
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--verify", action="store_true", help="Verify export with test inference")
    args = parser.parse_args()

    print("=" * 60)
    print("ONNX Model Export")
    print("=" * 60)

    # Determine output path
    if args.output is None:
        model_path = Path(args.model)
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    else:
        output_path = Path(args.output)

    # Create exporter
    print(f"\nLoading model from: {args.model}")
    exporter = PolicyExporter(args.model)

    # Print model info
    info = exporter.get_model_info()
    print(f"\nModel Information:")
    print(f"  Algorithm: {info['algorithm'].upper()}")
    print(f"  Observation shape: {info['observation_shape']}")
    print(f"  Action shape: {info['action_shape']}")
    print(f"  Total parameters: {info['total_parameters']:,}")

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    onnx_path = exporter.export_onnx(
        str(output_path),
        simplify=True
    )

    # Verify with test inference
    if args.verify:
        print("\n" + "-" * 40)
        print("Running verification inference...")

        # Load ONNX model
        engine = OnnxInferenceEngine(onnx_path)

        # Generate random test observation
        obs = np.random.randn(*info['observation_shape']).astype(np.float32)

        # Run inference
        action = engine.predict(obs)
        print(f"\n  Test observation shape: {obs.shape}")
        print(f"  Output action shape: {action.shape}")
        print(f"  Output action: {action}")

    print("\n" + "=" * 60)
    print(f"Export complete: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
