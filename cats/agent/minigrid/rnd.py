from torch import nn
from kitten.intrinsic.rnd import RandomNetworkDistillation


def build_rnd() -> RandomNetworkDistillation:
    def build_net():
        return nn.Sequential(
            nn.Conv2d(3, 8, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.LazyLinear(128),
        )

    return RandomNetworkDistillation(
        build_net(), build_net(), reward_normalisation=True
    )
