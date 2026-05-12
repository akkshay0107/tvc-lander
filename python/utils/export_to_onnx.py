import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.policy import PolicyNet


def export_to_onnx(model, input_path, output_path, obs_dim, n_frames):
    model.load_state_dict(torch.load(input_path))
    model.eval()

    # ONNX needs a dummy input to trace through the model
    # Shape should be (batch, obs_dim * n_frames)
    dummy_input = torch.randn(1, obs_dim * n_frames)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output_mean", "output_std"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output_mean": {0: "batch_size"},
            "output_std": {0: "batch_size"},
        },
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
