import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, ReseedWrapper, PositionBonus

from kitten.common import util

from cats.env import (
    FixedResetWrapper,
    StochasticActionWrapper,
)
from cats.reset import ResetActionWrapper

# Offline, continuous,
classic_control = ["MountainCarContinuous-v0", "Pendulum-v1", "MountainCar-v0", "CartPole-v1", "Acrobot-v1", "HalfCheetah-v4"]

# Minigrid
minigrid = ["MiniGrid-FourRooms-v0"]


def build_env(cfg) -> gym.Env:
    if cfg.env.name in classic_control:
        env = util.build_env(**cfg.env)
        if cfg.cats.fixed_reset:
            env = FixedResetWrapper(env)
        # TODO: Stochastic actions (aleatoric uncertainty)
        # if self.environment_action_noise > 0:
        #     self.env = StochasticActionWrapper(
        #         self.env, noise=self.environment_action_noise
        #     )
    elif cfg.env.name in minigrid:
        env = util.build_env(**cfg.env)
        # Convert to MDP
        env = FullyObsWrapper(env)
        #env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        if cfg.cats.fixed_reset:
            env = ReseedWrapper(env, seeds=(int(cfg.seed),))
    else:
        raise ValueError("Unknown environment")
    if cfg.cats.reset_action.enable:
        env = ResetActionWrapper(env, penalty=cfg.cats.reset_action.penalty, deterministic=True)
    return env
