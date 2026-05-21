# tvc-lander

![Demo](assets/demo_landing_new.gif)

A Proximal Policy Optimization (PPO) agent trained to land a rocket using thrust vector controls. The gym environment is built in Rust using `macroquad` and `rapier2d` and is inspired by the LunarLander task. The landing task has been modified to have continuous control instead of a discrete action space and to be a bit simpler (fixed landing zone, flat land).

## Live Demo

A live demo of the PPO agent is available [here](https://akkshay0107.github.io/tvc-lander/). The demo is a WASM build of the `controlled_sim` binary. The model weights are included in the binary and run in Rust using ONNX. The model kinda works now (more detail below in the results section). You can click and drag the rocket to any position on the screen to see how the agent recovers.

## Features

This project is split into several distinct parts that work together to create the final simulation:

*   **`base` (Rust)**: The core physics engine. It uses `rapier2d` for rigid-body dynamics and `macroquad` for hardware-accelerated rendering. It handles the rocket's thrust physics, gimbal limits, and collision detection.
*   **`gym` (Rust/Python)**: An FFI binding crate using `pyo3`. It wraps the Rust simulation into a Python module that can be accessed for RL training. It supports vectorized environments, allowing the training script to run 64+ simulations in parallel using `rayon`.
*   **`python` (PyTorch)**: The reinforcement learning section. It implements a PPO agent with frame stacking (4 frames) and a curriculum learning manager that gradually increases the difficulty of the landing task.
*   **Inference (Tract)**: Once trained, the PyTorch model is exported to ONNX. The Rust simulation loads this model using `tract-onnx` which lets it run in WASM environments without external dependencies.

## Results

Table below shows the general distribution (from `python/tests/test_model.py`) of how the model lands a rocket over 1000 episodes.

| Status | Count | Percentage |
| :--- | :--- | :--- |
| Success | 961 | 96.10% |
| Crash | 36 | 3.60% |
| Missing Target (Landed but outside the zone) | 3 | 0.30% |

On manually testing it, the model has been able to leverage some oddities in the environment to get such a high success rate. For example, the environment code terminates an episode if the COM of the rocket is sufficiently low (slightly above an upright rocket at rest). The policy manages to tilt the rocket on landing to avoid episode termination and drag itself along the ground into the landing zone before fixing its orientation. Fixing this reward hacking would probably require some better landing detection, but the model seems to work for reasonable inputs. 

These results also took a bit of resetting with multiple different random seeds. Some of the training runs often got stuck at a low curriculum level and deteriorated over time. 

## Technical Details

### Observation Space
The agent receives a 6-dimensional state vector (normalized roughly to $[-1, 1]$):
- $[x, y]$ position
- $\theta$ (angle from vertical)
- $[v_x, v_y]$ linear velocity
- $\omega$ angular velocity

### Action Space
The agent controls two continuous parameters (also normalized to the range $[-1, 1]$):
- Thrust
- Gimbal Angle: The nozzle direction for thrust vectoring.

### Model Architecture
The PPO agent uses two separate neural networks for the actor (policy) and critic (value function). Frame stacking is used to give the model context over the last 4 frames. Both networks are simple MLPs (with LeakyReLU as the activation function and LayerNorm). The policy network outputs actions using a tanh squashed gaussian (which is then scaled by the gym environment to the actual physical amount). 

The above architecture was reached after trying out ideas with different activation functions / no frame stacking / shared backbone for policy and value nets which didn't converge as quickly (or work as well).

Another modification in the training loop is the noise schedule, where a fixed noise sampled per environment every 120 steps (2s at 60Hz) is added to the output of the policy to encourage exploration (as opposed to every step in PPO). This causes correlation between the noise in each time step and increases the variance in the updates, but the upside is more consistent exploration without the jitter that the regular PPO noise sampling does at high frequencies. 

The above modification was done after reading through the gSDE paper and trying that out. Unfortunately, the gSDE idea wasn't working too well with large entropy swings between different states, but I liked the idea of smoother noisy trajectories which is what I kept (removing the state dependency).  

### Curriculum Learning
Training starts with the rocket hovering directly above the target and slowly expands to:
1.  Fixed vertical drops.
2.  Randomized initial angles.
3.  Randomized horizontal offsets.
4.  Full range of motion and high-altitude starts.

## Getting Started

### Prerequisites
- **Rust**: Latest stable version.
- **Python 3.12+**: Managed via `uv` or `poetry`.
- **wasm-bindgen CLI**: Only if you want to build the web demo locally.

### Installation

1.  Clone the repo:
    ```bash
    git clone https://github.com/akkshay0107/tvc-lander.git
    cd tvc-lander
    ```
2.  Set up the Python environment:
    ```bash
    cd python
    uv sync
    ```

## Workflow

### 1. Training the Agent
Build the gym module and start the training loop:
```bash
# In the python directory
maturin develop --release
python src/ppo.py
```
Training progress is logged to `training.log`. Checkpoints are saved to `python/models/`.

### 2. Exporting the Model
Convert the PyTorch `.pth` weights to an ONNX graph:
```bash
python utils/export_to_onnx.py
```

### 3. Running the Simulation
Run the interactive Rust simulation using the exported model:
```bash
cargo run --bin controlled_sim --release
```

### 4. Building for Web
The project includes a helper script to bundle the simulation for WASM:
```bash
./build_wasm.sh controlled_sim --release
```
The output will be in the `dist/` directory. You can then serve it using any http server module.
