import logging
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
        lr=5e-6,
        epochs=10,
        batch_size=1024,
        ent_coef=0.01,
        target_kl=0.015,
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

    @torch.inference_mode()
    def collect_trajectories(self, horizon=2048):
        group_size = self.env.group_size
        obs = self.env.reset()

        # per rollout setup
        frame_buffers = [deque(maxlen=self.n_frames) for _ in range(group_size)]
        stacked_obs = self._get_stacked_obs(obs, frame_buffers)

        # per agent buffers
        obs_bufs = [[] for _ in range(group_size)]
        act_bufs = [[] for _ in range(group_size)]
        logp_bufs = [[] for _ in range(group_size)]
        rew_bufs = [[] for _ in range(group_size)]
        done_bufs = [[] for _ in range(group_size)]
        trunc_bufs = [[] for _ in range(group_size)]
        val_bufs = [[] for _ in range(group_size)]
        val_next_bufs = [[] for _ in range(group_size)]

        agent_dones = [False] * group_size
        stats = {
            "success": 0,
            "crash": 0,
            "out_of_bounds": 0,
            "timeout": 0,
            "missing_target": 0,
        }

        for t in range(horizon):
            if all(agent_dones):
                break

            # noise sampled every 2s assuming 60 Hz
            # middle ground between rollout level bias and jitter
            if t % 120 == 0:
                self.policy.sample_noise(group_size)

            obs_tensor = torch.as_tensor(
                stacked_obs, dtype=torch.float32, device=self.device
            )
            dist = self.policy.get_dist(obs_tensor)
            values = self.value(obs_tensor)

            raw_actions = self.policy(obs_tensor)
            actions = torch.tanh(raw_actions)

            eps = 1e-6
            jac_term = torch.log(1.0 - actions.pow(2) + eps)
            logps = (dist.log_prob(raw_actions) - jac_term).sum(dim=-1)

            actions_np = actions.cpu().numpy().astype(np.float32)
            next_obs, rewards, next_dones, reasons = self.env.step(actions_np.tolist())

            next_stacked_obs = self._get_stacked_obs(next_obs, frame_buffers)
            next_obs_tensor = torch.as_tensor(
                next_stacked_obs, dtype=torch.float32, device=self.device
            )
            next_values = self.value(next_obs_tensor)

            for i in range(group_size):
                if not agent_dones[i]:
                    obs_bufs[i].append(stacked_obs[i])
                    act_bufs[i].append(raw_actions[i].cpu().numpy())
                    logp_bufs[i].append(logps[i].item())
                    rew_bufs[i].append(rewards[i])
                    done_bufs[i].append(next_dones[i])
                    is_trunc = next_dones[i] and reasons[i] == "timeout"
                    trunc_bufs[i].append(is_trunc)
                    val_bufs[i].append(values[i].item())
                    val_next_bufs[i].append(next_values[i].item())

                    if next_dones[i]:
                        agent_dones[i] = True
                        if reasons[i] in stats:
                            stats[reasons[i]] += 1

            stacked_obs = next_stacked_obs

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
            stats,
            np.mean([len(b) for b in obs_bufs if b]),
        )

    def train(
        self,
        curriculum: CurriculumManager,
        num_rollouts: int,
        horizon: int = 2048,
        start_level: int = 0,
    ) -> None:
        if 0 <= start_level < len(curriculum.tasks):
            curriculum._task_idx = start_level

        agg_stats = {
            "pi_loss": [],
            "v_loss": [],
            "entropy": [],
            "kl": [],
            "ep_len": [],
            "success": 0,
            "crash": 0,
            "out_of_bounds": 0,
            "timeout": 0,
            "missing_target": 0,
            "total_episodes": 0,
        }

        for rollout in range(1, num_rollouts + 1):
            obs_b, act_b, logp_old_b, adv_b, ret_b, stats, avg_ep_len = (
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
            r_pi_loss, r_v_loss, r_ent, r_kl, r_batches = 0.0, 0.0, 0.0, 0.0, 0

            for _ in range(self.epochs):
                early_stop = False
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, self.batch_size):
                    b = idx[start : start + self.batch_size]
                    obs_mb, raw_act_mb = obs_t[b], act_t[b]
                    logp_old_mb, adv_mb, ret_mb = logp_old_t[b], adv_t[b], ret_t[b]

                    dist = self.policy.get_dist(obs_mb)
                    act_mb = torch.tanh(raw_act_mb)

                    eps = 1e-6
                    jac_term = torch.log(1.0 - act_mb.pow(2) + eps)
                    logp = (dist.log_prob(raw_act_mb) - jac_term).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)

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
                        # stable approx of kl
                        logr = logp - logp_old_mb
                        kl = ((logr.exp() - 1) - logr).mean()

                    r_pi_loss += pi_loss.item()
                    r_v_loss += value_loss.item()
                    r_ent += entropy.mean().item()
                    r_kl += kl.item()
                    r_batches += 1

                    if (
                        self.target_kl is not None
                        and (r_kl / r_batches) > self.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break

            agg_stats["pi_loss"].append(r_pi_loss / r_batches)
            agg_stats["v_loss"].append(r_v_loss / r_batches)
            agg_stats["entropy"].append(r_ent / r_batches)
            agg_stats["kl"].append(r_kl / r_batches)
            agg_stats["ep_len"].append(avg_ep_len)
            agg_stats["total_episodes"] += self.env.group_size
            for k in ["success", "crash", "out_of_bounds", "timeout", "missing_target"]:
                agg_stats[k] += stats[k]

            # checkpointing + val loop
            if rollout % 100 == 0:
                val_sr = self.validate(num_rollouts=4, horizon=horizon)
                logging.info(f"Validation: Rollout {rollout}, SR: {val_sr:.2%}")

                prev_idx = curriculum.task_idx
                if val_sr >= 0.85:
                    if curriculum.step_up():
                        logging.info(f"[Curriculum] UP to Lvl {curriculum.task_idx}")
                elif val_sr < 0.20:
                    if curriculum.step_down():
                        logging.info(f"[Curriculum] DOWN to Lvl {curriculum.task_idx}")

                # auto save
                torch.save(self.policy.state_dict(), "./models/policy_net.pth")
                torch.save(self.value.state_dict(), "./models/value_net.pth")

                # level up save
                if prev_idx < curriculum.task_idx:
                    torch.save(
                        self.policy.state_dict(),
                        f"./models/policy_lvl_{prev_idx}.pth",
                    )

            # aggregate stats from last 10 rollouts
            if rollout % 10 == 0:
                avg_pi = np.mean(agg_stats["pi_loss"])
                avg_v = np.mean(agg_stats["v_loss"])
                avg_ent = np.mean(agg_stats["entropy"])
                avg_kl = np.mean(agg_stats["kl"])
                avg_len = np.mean(agg_stats["ep_len"])
                total = agg_stats["total_episodes"]
                freq = {
                    k: (agg_stats[k] / total) * 100
                    for k in [
                        "success",
                        "crash",
                        "out_of_bounds",
                        "timeout",
                        "missing_target",
                    ]
                }
                log_msg = (
                    f"Rollout {rollout:5d} [Lvl {curriculum.task_idx}] | "
                    f"Success: {freq['success']:5.1f}% | Crash: {freq['crash']:4.1f}% | "
                    f"OOB: {freq['out_of_bounds']:4.1f}% | Time: {freq['timeout']:4.1f}% | "
                    f"Miss: {freq['missing_target']:4.1f}%\n"
                    f"      Losses: Pi {avg_pi:8.4f}, V {avg_v:8.4f}, Ent {avg_ent:5.2f}, KL {avg_kl:8.5f} | "
                    f"AvgLen: {avg_len:4.0f}"
                )
                logging.info(log_msg)
                for k in ["pi_loss", "v_loss", "entropy", "kl", "ep_len"]:
                    agg_stats[k] = []
                for k in [
                    "success",
                    "crash",
                    "out_of_bounds",
                    "timeout",
                    "missing_target",
                    "total_episodes",
                ]:
                    agg_stats[k] = 0

    @torch.inference_mode()
    def validate(self, num_rollouts=4, horizon=2048):
        group_size = self.env.group_size
        total_success, total_ep = 0, 0
        for _ in range(num_rollouts):
            obs = self.env.reset()
            frame_bufs = [deque(maxlen=self.n_frames) for _ in range(group_size)]
            stacked_obs = self._get_stacked_obs(obs, frame_bufs)
            dones = [False] * group_size
            for _ in range(horizon):
                if all(dones):
                    break
                obs_t = torch.as_tensor(stacked_obs, device=self.device)
                acts = torch.tanh(self.policy(obs_t, deterministic=True))
                next_obs, _, next_dones, reasons = self.env.step(
                    acts.cpu().numpy().tolist()
                )
                stacked_obs = self._get_stacked_obs(next_obs, frame_bufs)
                for i in range(group_size):
                    if not dones[i] and next_dones[i]:
                        dones[i] = True
                        total_ep += 1
                        if reasons[i] == "success":
                            total_success += 1
        return total_success / total_ep if total_ep > 0 else 0.0


def main():
    os.makedirs("./models", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training.log", mode="w"),
        ],
    )

    env = PyEnvironment(max_steps=4096, group_size=64)
    tasks = [
        BoxBound(40.0, 40.0, 5.0, 10.0, 0.0, 0.0),
        BoxBound(40.0, 40.0, 10.0, 20.0, -0.1, 0.1),
        BoxBound(30.0, 50.0, 20.0, 30.0, -0.3, 0.3),
        BoxBound(20.0, 60.0, 30.0, 40.0, -0.1, 0.1),
        BoxBound(5.0, 75.0, 30.0, 40.0, -0.1, 0.1),
        BoxBound(20.0, 60.0, 30.0, 40.0, -0.3, 0.3),
        BoxBound(5.0, 75.0, 30.0, 40.0, -0.3, 0.3),
    ]
    curriculum = CurriculumManager(env, tasks)
    agent = PPOAgent(env, n_frames=4)

    # load old checkpoint if exists
    model_path = "./models/policy_net.pth"
    if os.path.exists(model_path):
        logging.info(f"Loading model from {model_path}")
        agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
        # force reset entropy (to around -1.2)
        # for the case when entropy collapses / explodes
        # but still want to reuse model (delete old file otherwise)
        with torch.no_grad():
            agent.policy.log_std.fill_(-2.0)

    logging.info(
        f"Training started! ({env.group_size} parallel envs, {agent.n_frames} frames stacked)"
    )
    agent.train(curriculum, num_rollouts=20_000)


if __name__ == "__main__":
    main()
