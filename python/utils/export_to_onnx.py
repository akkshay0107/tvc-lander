import sys
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.policy import PolicyNet


class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)

        latent = self.policy.feature_extractor(obs)
        mean = self.policy.mean_layer(latent)
        return mean


def export_to_onnx(model, input_path, output_path, obs_dim, n_frames):
    device = torch.device("cpu")
    checkpoint = torch.load(input_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    wrapper = PolicyWrapper(model)
    # ONNX needs a dummy input to trace through the model
    # Shape should be (batch, obs_dim * n_frames)
    dummy_input = torch.randn(1, obs_dim * n_frames)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output_mean"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output_mean": {0: "batch_size"},
        },
        opset_version=12,
    )
    print("Export complete.")


if __name__ == "__main__":
    # Exporting only the policy net for generating actions
    OBS_DIM = 6
    ACT_DIM = 2
    N_FRAMES = 4
    model = PolicyNet(OBS_DIM, ACT_DIM, n_frames=N_FRAMES)
    export_to_onnx(
        model, "./models/policy_net.pth", "./models/policy_net.onnx", OBS_DIM, N_FRAMES
    )
