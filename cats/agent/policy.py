
import copy
import gymnasium as gym

from kitten.common.rng import Generator
from kitten.policy import Policy

class ExplorationPolicy(Policy):
    """Purely Random Exploration with probability of repeat"""

    def __init__(
        self, env: gym.Env, rng: Generator, repeat_probability: float = 0.9
    ) -> None:
        super().__init__(fn=None)
        self.action_space = copy.deepcopy(env.action_space)
        self.action_space.seed(int(rng.numpy.integers(2**32 - 1)))
        self.rng = rng
        self.p = repeat_probability
        self.previous_action = None

    def __call__(self, obs):
        if self.previous_action is None or self.rng.numpy.random() > self.p:
            self.previous_action = self.action_space.sample()
            return self.previous_action
        else:
            return self.previous_action

    def reset(self) -> None:
        self.previous_action = None