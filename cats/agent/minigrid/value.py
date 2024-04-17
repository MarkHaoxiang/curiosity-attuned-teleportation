from torch import nn, Tensor
from kitten.nn import Value, HasValue


# Define Value Module, Transformations, Policy
class MinigridValue(Value, HasValue):

    def __init__(self) -> None:
        super().__init__()
        # We use the CNN from https://minigrid.farama.org/content/training/
        # With ReLu changed to LeakyReLu

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        # Followed by a simple MLP
        self.linear = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(1),
        )

    def forward(self, x) -> Tensor:
        return self.linear(self.cnn(x))

    @property
    def value(self) -> Value:
        return self

    def v(self, s: Tensor) -> Tensor:
        d = len(s.shape) == 3
        if d:
            s = s.unsqueeze(0)
        if s.shape[-1] == 3:
            s = s.permute(0, 3, 1, 2)
        v = self.forward(s).squeeze()
        return v
