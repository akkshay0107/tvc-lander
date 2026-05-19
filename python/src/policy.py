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

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # SDE noise buffers
        self.register_buffer("sde_noise_weights", torch.empty(0), persistent=False)

        gain = nn.init.calculate_gain("leaky_relu")
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                _init_layer(layer, gain)

        _init_layer(self.mean_layer, 0.1)

    def sample_noise(self, batch_size: int):
        device = self.log_std.device
        # state dependent noise
        self.sde_noise_weights = torch.randn(
            batch_size, self.hidden_dim, self.action_dim, device=device
        )

    def get_noise(self, latent_features: torch.Tensor) -> torch.Tensor:
        # if noise hasn't been sampled => deterministic eval
        if (
            self.sde_noise_weights is None
            or self.sde_noise_weights.dim() == 0
            or self.sde_noise_weights.shape[0] != latent_features.shape[0]
        ):
            return torch.zeros(
                latent_features.shape[0], self.action_dim, device=latent_features.device
            )

        norm_latent = latent_features / (self.hidden_dim**0.5)  # unit var
        norm_latent = norm_latent.unsqueeze(1)  # (B, 1, H)
        noise = torch.bmm(norm_latent, self.sde_noise_weights).squeeze(1)
        # (B, 1, H) dot (B, H, A) = (B, 1, A) -> (B, A) after squeeze
        return noise * self.log_std.exp()

    def forward(self, observations: torch.Tensor, deterministic: bool = False):
        if observations.dim() == 3:
            observations = observations.view(observations.size(0), -1)

        latent_features = self.feature_extractor(observations)
        mean_actions = self.mean_layer(latent_features)

        if deterministic:
            return mean_actions, None

        noise = self.get_noise(latent_features)
        return mean_actions + noise, latent_features

    def get_dist(self, observations: torch.Tensor):
        if observations.dim() == 3:
            observations = observations.view(observations.size(0), -1)

        latent_features = self.feature_extractor(observations)
        mean = self.mean_layer(latent_features)
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
