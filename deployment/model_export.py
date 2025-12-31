"""
Model Export Utilities

Export trained RL models to various formats for deployment:
- ONNX for cross-platform inference
- TorchScript for C++ deployment
- SavedModel for TensorFlow Serving
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

import torch
import torch.nn as nn


class PolicyExporter:
    """
    Export trained policies to deployment-ready formats.

    Supports:
    - ONNX export for universal deployment
    - TorchScript export for C++/embedded
    - Standalone PyTorch model extraction
    """

    def __init__(self, model_path: str, algorithm: str = "auto"):
        """
        Initialize exporter with a trained model.

        Args:
            model_path: Path to trained SB3 model (.zip)
            algorithm: Algorithm type ('ppo', 'sac', 'td3', 'auto')
        """
        self.model_path = Path(model_path)
        self.algorithm = algorithm

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.sb3_model = None
        self.policy = None
        self._load_model()

    def _load_model(self):
        """Load the SB3 model and extract policy."""
        from stable_baselines3 import PPO, SAC, TD3

        algorithms = {"ppo": PPO, "sac": SAC, "td3": TD3}

        if self.algorithm == "auto":
            # Try each algorithm
            for name, AlgoClass in algorithms.items():
                try:
                    self.sb3_model = AlgoClass.load(str(self.model_path))
                    self.algorithm = name
                    break
                except Exception:
                    continue

            if self.sb3_model is None:
                raise ValueError(f"Could not load model from {self.model_path}")
        else:
            AlgoClass = algorithms.get(self.algorithm.lower())
            if AlgoClass is None:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            self.sb3_model = AlgoClass.load(str(self.model_path))

        # Extract policy network
        self.policy = self.sb3_model.policy

        print(f"Loaded {self.algorithm.upper()} model from {self.model_path}")

    def export_onnx(
        self,
        output_path: str,
        observation_shape: Optional[Tuple[int, ...]] = None,
        opset_version: int = 11,
        dynamic_axes: bool = True,
        simplify: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path for output .onnx file
            observation_shape: Shape of observations (auto-detected if None)
            opset_version: ONNX opset version
            dynamic_axes: Allow dynamic batch size
            simplify: Run ONNX simplifier (requires onnxsim)

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get observation shape
        if observation_shape is None:
            obs_space = self.sb3_model.observation_space
            observation_shape = obs_space.shape

        # Create wrapper for clean export
        policy_wrapper = OnnxPolicyWrapper(self.policy, self.algorithm)

        # Create dummy input
        dummy_input = torch.randn(1, *observation_shape)

        # Export configuration
        input_names = ["observation"]
        output_names = ["action"]

        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            }

        # Export to ONNX
        torch.onnx.export(
            policy_wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
        )

        print(f"Exported ONNX model to: {output_path}")

        # Optionally simplify
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model = onnx.load(str(output_path))
                model_simplified, check = onnx_simplify(model)

                if check:
                    onnx.save(model_simplified, str(output_path))
                    print("ONNX model simplified successfully")
                else:
                    print("Warning: ONNX simplification check failed")

            except ImportError:
                print("Note: Install onnxsim for model simplification")

        # Verify export
        self._verify_onnx(output_path, observation_shape)

        return str(output_path)

    def _verify_onnx(self, onnx_path: Path, observation_shape: Tuple[int, ...]):
        """Verify ONNX export produces matching outputs."""
        try:
            import onnxruntime as ort

            # Load ONNX model
            session = ort.InferenceSession(str(onnx_path))

            # Create test input
            test_obs = np.random.randn(1, *observation_shape).astype(np.float32)

            # Run ONNX inference
            onnx_input = {session.get_inputs()[0].name: test_obs}
            onnx_output = session.run(None, onnx_input)[0]

            # Run PyTorch inference
            with torch.no_grad():
                torch_input = torch.from_numpy(test_obs)
                torch_output, _, _ = self.policy(torch_input, deterministic=True)
                torch_output = torch_output.numpy()

            # Compare outputs
            max_diff = np.max(np.abs(onnx_output - torch_output))
            print(f"ONNX verification: max output difference = {max_diff:.6f}")

            if max_diff < 1e-5:
                print("ONNX export verified successfully!")
            else:
                print("Warning: ONNX outputs differ from PyTorch")

        except ImportError:
            print("Note: Install onnxruntime to verify ONNX export")

    def export_torchscript(
        self,
        output_path: str,
        method: str = "trace",
        observation_shape: Optional[Tuple[int, ...]] = None,
    ) -> str:
        """
        Export model to TorchScript format.

        Args:
            output_path: Path for output .pt file
            method: 'trace' or 'script'
            observation_shape: Shape of observations

        Returns:
            Path to exported TorchScript file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get observation shape
        if observation_shape is None:
            obs_space = self.sb3_model.observation_space
            observation_shape = obs_space.shape

        # Create wrapper
        policy_wrapper = TorchScriptPolicyWrapper(self.policy, self.algorithm)
        policy_wrapper.eval()

        if method == "trace":
            # Trace-based export
            dummy_input = torch.randn(1, *observation_shape)
            scripted_model = torch.jit.trace(policy_wrapper, dummy_input)
        else:
            # Script-based export
            scripted_model = torch.jit.script(policy_wrapper)

        scripted_model.save(str(output_path))
        print(f"Exported TorchScript model to: {output_path}")

        return str(output_path)

    def export_pytorch(self, output_path: str) -> str:
        """
        Export raw PyTorch state dict.

        Args:
            output_path: Path for output .pth file

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save policy state dict
        state_dict = self.policy.state_dict()
        torch.save(state_dict, str(output_path))

        print(f"Exported PyTorch state dict to: {output_path}")
        return str(output_path)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        obs_space = self.sb3_model.observation_space
        act_space = self.sb3_model.action_space

        # Count parameters
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )

        return {
            "algorithm": self.algorithm,
            "observation_shape": obs_space.shape,
            "action_shape": act_space.shape,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "policy_class": self.policy.__class__.__name__,
        }


class OnnxPolicyWrapper(nn.Module):
    """Wrapper for ONNX export that outputs deterministic actions."""

    def __init__(self, policy, algorithm: str):
        super().__init__()
        self.policy = policy
        self.algorithm = algorithm

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass returning deterministic action."""
        # Get features
        features = self.policy.extract_features(observation)

        if hasattr(self.policy, 'mlp_extractor'):
            # PPO/A2C style
            latent_pi, _ = self.policy.mlp_extractor(features)
            mean_actions = self.policy.action_net(latent_pi)
        else:
            # SAC/TD3 style
            mean_actions = self.policy.actor(features)

        return mean_actions


class TorchScriptPolicyWrapper(nn.Module):
    """Wrapper for TorchScript export."""

    def __init__(self, policy, algorithm: str):
        super().__init__()
        self.features_extractor = policy.features_extractor

        if hasattr(policy, 'mlp_extractor'):
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net
            self.style = "ppo"
        else:
            self.actor = policy.actor
            self.style = "sac"

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.features_extractor(observation)

        if self.style == "ppo":
            latent_pi, _ = self.mlp_extractor(features)
            return self.action_net(latent_pi)
        else:
            return self.actor(features)


def export_model_to_onnx(
    model_path: str,
    output_path: str,
    **kwargs
) -> str:
    """
    Convenience function to export model to ONNX.

    Args:
        model_path: Path to trained model
        output_path: Path for output ONNX file
        **kwargs: Additional arguments for export

    Returns:
        Path to exported file
    """
    exporter = PolicyExporter(model_path)
    return exporter.export_onnx(output_path, **kwargs)


def export_model_to_torchscript(
    model_path: str,
    output_path: str,
    **kwargs
) -> str:
    """
    Convenience function to export model to TorchScript.

    Args:
        model_path: Path to trained model
        output_path: Path for output TorchScript file
        **kwargs: Additional arguments for export

    Returns:
        Path to exported file
    """
    exporter = PolicyExporter(model_path)
    return exporter.export_torchscript(output_path, **kwargs)


class OnnxInferenceEngine:
    """
    Lightweight ONNX inference engine for deployment.

    Provides fast inference without SB3/PyTorch dependencies.
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        """
        Initialize ONNX inference engine.

        Args:
            onnx_path: Path to ONNX model
            device: 'cpu' or 'cuda'
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Install onnxruntime: pip install onnxruntime")

        # Select execution provider
        if device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.observation_shape = tuple(input_shape[1:])

        print(f"Loaded ONNX model from: {onnx_path}")
        print(f"  Input shape: {self.observation_shape}")
        print(f"  Device: {device}")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Run inference on observation.

        Args:
            observation: Input observation
            deterministic: Ignored (ONNX always deterministic)

        Returns:
            Action array
        """
        # Ensure correct shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Ensure float32
        observation = observation.astype(np.float32)

        # Run inference
        action = self.session.run(
            [self.output_name],
            {self.input_name: observation}
        )[0]

        return action

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Shorthand for predict."""
        return self.predict(observation)
