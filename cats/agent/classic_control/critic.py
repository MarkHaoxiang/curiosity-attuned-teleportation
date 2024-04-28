import torch
from torch import Tensor
import torch.nn as nn
import gymnasium as gym

from kitten.nn import Critic

from cats.reset import ResetActionWrapper
from cats.reset import ResetMemory

class ClassicalResetCritic(Critic):
    def __init__(self,
                 env: gym.Env,
                 features: int = 128
    ) -> None:
        super().__init__()
        assert isinstance(env, ResetActionWrapper), "Environment must have reset action"
        assert isinstance(env.action_space, gym.spaces.Box), "Use ClassicalDiscreteCritic"
        assert env._deterministic, "Not yet implemented"
        self.env = env
        self.net = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[-1]
                    + env.action_space.shape[0] -1,
                    out_features=features,
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=2), # First index is non-truncate, second is truncate
        )

    def q(self, s: Tensor, a: Tensor):
        a_ = a[..., :-1]
        t = a[..., -1]
        combined = torch.cat((s,a_), dim=-1)
        heads = self(combined)
        q = torch.where(t<=0.5, heads[..., 0], heads[..., 1] - t)
        return q 

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class ClassicalInjectionResetCritic(Critic):
    def __init__(self,
                 env: gym.Env,
                 features: int = 128,
    ) -> None:
        super().__init__()
        assert isinstance(env, ResetActionWrapper), "Environment must have reset action"
        assert isinstance(env.action_space, gym.spaces.Box), "Not yet supported"
        assert env._deterministic, "Not yet implemented"


        self.env = env
        # Just a standard critic for the unwrapped enviroment
        self.net = nn.Sequential(
                nn.Linear(
                    in_features=env.observation_space.shape[-1]
                    + env.action_space.shape[0] -1,
                    out_features=features,
                ),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=features),
                nn.LeakyReLU(),
                nn.Linear(in_features=features, out_features=2),
        )
        self.reset_value = 0

    def q(self, s: Tensor, a: Tensor):
        a_ = a[..., :-1]
        t = a[..., -1]
        combined = torch.cat((s,a_), dim=-1)
        heads = self(combined)
        q = torch.where(
            t<=0.5,
            heads[..., 0],
            torch.clamp(heads[..., 1], min=0) + 
            self.reset_value - t
        )
        return q 


    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
