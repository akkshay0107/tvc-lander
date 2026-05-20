import os
import sys
from collections import Counter, deque

import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from gym import PyEnvironment
from policy import PolicyNet


def run_performance_test(model_path, num_episodes=1000, group_size=32, n_frames=4):
    env = PyEnvironment(max_steps=4096, group_size=group_size)

    # BoxBound(5.0, 75.0, 30.0, 40.0, -0.3, 0.3)
    env.set_spawn_x_range(5.0, 75.0)
    env.set_spawn_y_range(30.0, 40.0)
    env.set_spawn_angle_range(-0.3, 0.3)

    print(f"Loading model from {model_path}...")
    device = "cpu"
    policy = PolicyNet(env.obs_dim, env.act_dim, n_frames=n_frames).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    policy.eval()

    reasons_counter = Counter()
    episodes_completed = 0

    print(f"Starting {num_episodes} simulations with group_size={group_size}...")

    while episodes_completed < num_episodes:
        obs = env.reset()
        # Store starting positions for each agent in the group
        start_positions = [(o[0], o[1]) for o in obs]

        frame_buffers = [deque(maxlen=n_frames) for _ in range(group_size)]

        def get_stacked_obs(obs_list):
            stacked = []
            for i, o in enumerate(obs_list):
                buffer = frame_buffers[i]
                if len(buffer) == 0:
                    for _ in range(n_frames):
                        buffer.append(o)
                else:
                    buffer.append(o)
                stacked.append(np.concatenate(list(buffer)))
            return np.array(stacked, dtype=np.float32)

        current_obs = get_stacked_obs(obs)
        agent_dones = [False] * group_size

        while not all(agent_dones) and episodes_completed < num_episodes:
            obs_tensor = torch.as_tensor(
                current_obs, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                mean = policy(obs_tensor, deterministic=True)
                actions = torch.tanh(mean)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, reasons = env.step(actions_np.tolist())

            current_obs = get_stacked_obs(next_obs)

            for i in range(group_size):
                if not agent_dones[i] and dones[i]:
                    agent_dones[i] = True
                    status = reasons[i]
                    start_x, start_y = start_positions[i]
                    end_x, end_y = next_obs[i][0], next_obs[i][1]

                    episodes_completed += 1
                    reasons_counter[status] += 1

                    print(
                        f"Episode {episodes_completed:4d}: Start({start_x:5.2f}, {start_y:5.2f}) -> "
                        f"End({end_x:5.2f}, {end_y:5.2f}) | Status: {status}"
                    )

                    if episodes_completed >= num_episodes:
                        break

    print("\n" + "=" * 50)
    print(f"PERFORMANCE TEST RESULTS ({episodes_completed} episodes)")
    print("=" * 50)
    print(f"{'Status':<25} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 50)

    for status, count in reasons_counter.most_common():
        percentage = (count / episodes_completed) * 100
        print(f"{status:<25} | {count:<8} | {percentage:>9.2f}%")

    print("=" * 50)


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(
        os.path.join(script_dir, "..", "models", "policy_net.pth")
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    run_performance_test(model_path)
