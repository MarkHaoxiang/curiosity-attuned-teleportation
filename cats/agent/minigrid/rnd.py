import torch
from torch import nn
from kitten.intrinsic.rnd import RandomNetworkDistillation

from .common import one_hot_encoding


def build_rnd() -> RandomNetworkDistillation:
    def build_net():
        return MinigridRND()

    return RandomNetworkDistillation(
        build_net(), build_net(), reward_normalisation=True
    )


class MinigridRND(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.LazyLinear(out_features=128),
        )

    def forward(self, x):
        image, direction = one_hot_encoding(x)
        lin_in = torch.cat((self.conv(image), direction), dim=1)
        # return self.linear(lin_in)
        return self.linear(lin_in)
