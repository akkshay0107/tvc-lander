import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from curriculum import BoxBound, CurriculumManager
from gym import PyEnvironment
from policy import PolicyNet, ValueNet


class PPOAgent:
    def __init__(
        self,
        env,
        n_frames=4,
        gamma=0.995,
        lam=0.95,
        clip_eps=0.2,
        lr=3e-4,
        epochs=10,
        batch_size=256,
        ent_coef=5e-3,
        target_kl=0.02,
        device="cpu",
    ):
        self.env = env
        self.n_frames = n_frames
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.target_kl = target_kl

        self.device = torch.device(device)
        self.policy = PolicyNet(
            self.env.obs_dim, self.env.act_dim, n_frames=n_frames
        ).to(self.device)
        self.value = ValueNet(self.env.obs_dim, n_frames=n_frames).to(self.device)
        self.optim = Adam(
            [
                {"params": self.policy.parameters(), "lr": lr},
                {"params": self.value.parameters(), "lr": lr},
            ]
        )

    def _compute_gae(self, rewards, values, next_values, bootstrap_mask, dones):
        T = rewards.shape[0]
        adv = torch.zeros((T,), device=rewards.device, dtype=rewards.dtype)
        gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(T)):
            delta = (
                rewards[t] + self.gamma * next_values[t] * bootstrap_mask[t] - values[t]
            )
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        return adv

    def _get_stacked_obs(self, obs_list, frame_buffers):
        stacked = []
        for i, obs in enumerate(obs_list):
            buffer = frame_buffers[i]
            if len(buffer) == 0:
                for _ in range(self.n_frames):
                    buffer.append(obs)
            else:
                buffer.append(obs)
            stacked.append(np.concatenate(list(buffer)))
        return np.array(stacked, dtype=np.float32)

    @torch.no_grad()
    def collect_trajectories(self, horizon=2048):
        group_size = self.env.group_size
        obs = self.env.reset()  # Vec of observations

        # Initialize frame buffers for stacking
        frame_buffers = [deque(maxlen=self.n_frames) for _ in range(group_size)]
        stacked_obs = self._get_stacked_obs(obs, frame_buffers)

        # Per-agent buffers
        obs_bufs = [[] for _ in range(group_size)]
        act_bufs = [[] for _ in range(group_size)]
        logp_bufs = [[] for _ in range(group_size)]
        rew_bufs = [[] for _ in range(group_size)]
        done_bufs = [[] for _ in range(group_size)]
        trunc_bufs = [[] for _ in range(group_size)]
        val_bufs = [[] for _ in range(group_size)]
        val_next_bufs = [[] for _ in range(group_size)]

        agent_dones = [False] * group_size
        success_count = 0

        for _ in range(horizon):
            if all(agent_dones):
                break

            obs_tensor = torch.as_tensor(
                stacked_obs, dtype=torch.float32, device=self.device
            )
            dist = self.policy.get_dist(obs_tensor)
            values = self.value(obs_tensor)

            actions = dist.rsample()
            logps = dist.log_prob(actions).sum(dim=-1)

            actions_np = actions.cpu().numpy().astype(np.float32)
            next_obs, rewards, next_dones, reasons = self.env.step(actions_np.tolist())

            # Prepare next stacked observation
            next_stacked_obs = self._get_stacked_obs(next_obs, frame_buffers)

            next_obs_tensor = torch.as_tensor(
                next_stacked_obs, dtype=torch.float32, device=self.device
            )
            next_values = self.value(next_obs_tensor)

            for i in range(group_size):
                if not agent_dones[i]:
                    obs_bufs[i].append(stacked_obs[i])
                    act_bufs[i].append(actions_np[i])
                    logp_bufs[i].append(logps[i].item())
                    rew_bufs[i].append(rewards[i])
                    done_bufs[i].append(next_dones[i])
                    is_trunc = next_dones[i] and reasons[i] == "timeout"
                    trunc_bufs[i].append(is_trunc)
                    val_bufs[i].append(values[i].item())
                    val_next_bufs[i].append(next_values[i].item())

                    if next_dones[i]:
                        agent_dones[i] = True
                        if reasons[i] == "success":
                            success_count += 1
                        # Reset frame buffer for this agent if it were to continue,
                        # but here we just stop collecting for it.

            stacked_obs = next_stacked_obs

        # Flatten all buffers
        flat_obs, flat_act, flat_logp, flat_adv, flat_ret = [], [], [], [], []

        for i in range(group_size):
            if not obs_bufs[i]:
                continue

            r = torch.tensor(rew_bufs[i], device=self.device)
            v = torch.tensor(val_bufs[i], device=self.device)
            vn = torch.tensor(val_next_bufs[i], device=self.device)
            d = torch.tensor(done_bufs[i], device=self.device)
            tr = torch.tensor(trunc_bufs[i], device=self.device)
            mask = 1.0 - (d.float() * (~tr).float())

            adv = self._compute_gae(r, v, vn, mask, d.float())
            ret = adv + v

            flat_obs.append(np.array(obs_bufs[i]))
            flat_act.append(np.array(act_bufs[i]))
            flat_logp.append(np.array(logp_bufs[i]))
            flat_adv.append(adv.cpu().numpy())
            flat_ret.append(ret.cpu().numpy())

        return (
            np.concatenate(flat_obs),
            np.concatenate(flat_act),
            np.concatenate(flat_logp),
            np.concatenate(flat_adv),
            np.concatenate(flat_ret),
            success_count,
        )

    def train(self, curriculum, num_rollouts, horizon=2048):
        for rollout in range(1, num_rollouts + 1):
            obs_b, act_b, logp_old_b, adv_b, ret_b, success_count = (
                self.collect_trajectories(horizon=horizon)
            )

            obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(act_b, dtype=torch.float32, device=self.device)
            logp_old_t = torch.as_tensor(
                logp_old_b, dtype=torch.float32, device=self.device
            )
            adv_t = torch.as_tensor(adv_b, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret_b, dtype=torch.float32, device=self.device)

            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n = obs_t.shape[0]
            avg_pi_loss, avg_v_loss, avg_entropy, avg_kl, num_batches = (
                0.0,
                0.0,
                0.0,
                0.0,
                0,
            )

            for epoch in range(self.epochs):
                early_stop = False
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.batch_size):
                    b = idx[start : start + self.batch_size]
                    obs_mb = obs_t[b]
                    act_mb = act_t[b]
                    logp_old_mb = logp_old_t[b]
                    adv_mb = adv_t[b]
                    ret_mb = ret_t[b]

                    dist = self.policy.get_dist(obs_mb)
                    logp = dist.log_prob(act_mb).sum(dim=-1)
                    entropy = dist.base_dist.entropy().sum(dim=-1)

                    ratio = torch.exp(logp - logp_old_mb)
                    surr1 = ratio * adv_mb
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * adv_mb
                    )
                    pi_loss = (
                        -(torch.min(surr1, surr2)).mean()
                        - self.ent_coef * entropy.mean()
                    )

                    value = self.value(obs_mb)
                    value_loss = nn.MSELoss()(value, ret_mb)

                    loss = pi_loss + 0.5 * value_loss

                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.policy.parameters()) + list(self.value.parameters()),
                        0.5,
                    )
                    self.optim.step()

                    with torch.no_grad():
                        kl = (logp_old_mb - logp).mean()

                    avg_pi_loss += pi_loss.item()
                    avg_v_loss += value_loss.item()
                    avg_entropy += entropy.mean().item()
                    avg_kl += kl.item()
                    num_batches += 1

                    if (
                        self.target_kl is not None
                        and (avg_kl / num_batches) > self.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break

            avg_pi_loss /= num_batches
            avg_v_loss /= num_batches
            avg_entropy /= num_batches
            avg_kl /= num_batches

            curriculum.update_metrics(success_count, self.env.group_size)

            print(
                f"Rollout: {rollout:3d} | Pi Loss: {avg_pi_loss:6.3f} | V Loss: {avg_v_loss:6.3f} | "
                f"Succ: {success_count:2d}/{self.env.group_size} | Lvl: {curriculum.task_idx} | "
                f"Entropy: {avg_entropy:5.2f} | KL: {avg_kl:6.4f}"
            )

            if rollout % 50 == 0:
                torch.save(self.policy.state_dict(), "./models/policy_net.pth")
                torch.save(self.value.state_dict(), "./models/value_net.pth")


def main():
    os.makedirs("./models", exist_ok=True)

    MAX_STEPS = 2000
    GROUP_SIZE = 32
    NUM_ROLLOUTS = 2000
    N_FRAMES = 4

    env = PyEnvironment(MAX_STEPS, GROUP_SIZE)

    tasks = [
        BoxBound(40.0, 40.0, 5.0, 10.0, 0.0, 0.0),
        BoxBound(40.0, 40.0, 10.0, 20.0, -0.1, 0.1),
        BoxBound(30.0, 50.0, 20.0, 30.0, -0.2, 0.2),
        BoxBound(20.0, 60.0, 30.0, 40.0, -0.3, 0.3),
        BoxBound(10.0, 70.0, 30.0, 40.0, -0.5, 0.5),
    ]
    curriculum = CurriculumManager(env, tasks)

    agent = PPOAgent(env, n_frames=N_FRAMES)

    print(
        f"Training started with {GROUP_SIZE} parallel agents and {N_FRAMES} frames stacking."
    )
    agent.train(curriculum, NUM_ROLLOUTS)
    print("Training completed.")


if __name__ == "__main__":
    main()
