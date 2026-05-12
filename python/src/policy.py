import torch
import torch.nn as nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def _init_layer(linear: nn.Linear, gain: float):
    nn.init.orthogonal_(linear.weight, gain)  # type: ignore
    nn.init.constant_(linear.bias, 0.0)


class PolicyNet(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_frames: int = 4
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_frames = n_frames

        input_dim = obs_dim * n_frames

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        gain = nn.init.calculate_gain("tanh")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

        _init_layer(self.mean_layer, 0.01)

    def forward(self, obs):
        # obs shape: (batch, n_frames * obs_dim) or (batch, n_frames, obs_dim)
        if obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)

        x = self.net(obs)
        mean = self.mean_layer(x)
        std = self.log_std.clamp(-20, 2).exp()
        return mean, std

    def get_dist(self, obs):
        mean, std = self.forward(obs)
        return TransformedDistribution(Normal(mean, std), [TanhTransform(cache_size=1)])


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_frames: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_frames = n_frames

        input_dim = obs_dim * n_frames

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        gain = nn.init.calculate_gain("tanh")
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)
        return self.net(obs).squeeze(-1)
