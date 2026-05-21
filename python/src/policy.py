import torch
import torch.nn as nn
from torch.distributions import Normal


def _init_layer(linear: nn.Linear, gain: float):
    nn.init.orthogonal_(linear.weight, gain)  # type: ignore
    nn.init.constant_(linear.bias, 0.0)


class PolicyNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_frames: int = 4,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_frames = n_frames
        self.hidden_dim = hidden_dim

        input_dim = observation_dim * n_frames

        self.actor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # noise per rollout
        self.register_buffer("noise", torch.empty(0), persistent=False)

        gain = nn.init.calculate_gain("leaky_relu")
        for layer in self.actor_net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

        _init_layer(self.mean_layer, 0.1)

    def sample_noise(self, batch_size: int):
        device = self.log_std.device
        # state dependent noise
        self.noise = torch.randn(batch_size, self.action_dim, device=device)

    def forward(self, observations: torch.Tensor, deterministic: bool = False):
        if observations.dim() == 3:
            observations = observations.view(observations.size(0), -1)

        z = self.actor_net(observations)
        mean = self.mean_layer(z)
        if deterministic:
            return mean

        return mean + self.log_std.exp() * self.noise

    def get_dist(self, observations: torch.Tensor):
        if observations.dim() == 3:
            observations = observations.view(observations.size(0), -1)

        z = self.actor_net(observations)
        mean = self.mean_layer(z)
        std = self.log_std.exp().expand_as(mean)

        return Normal(mean, std)


class ValueNet(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 128, n_frames: int = 4):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_frames = n_frames

        input_dim = observation_dim * n_frames

        self.critic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        gain = nn.init.calculate_gain("leaky_relu")
        for layer in self.critic_net:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

    def forward(self, observations: torch.Tensor):
        if observations.dim() == 3:
            observations = observations.view(observations.size(0), -1)
        return self.critic_net(observations).squeeze(-1)
