import copy
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from kitten.nn import HasValue
from kitten.policy import Policy
from kitten.common.rng import global_seed
from kitten.common.typing import Device
from kitten.logging import DictEngine, KittenLogger

from .env import build_env
from .cats import *


class ExperimentBase(ABC):
    """Contains a set of modules needed to run experiments"""

    def __init__(
        self,
        cfg: DictConfig,
        normalise_obs: bool = True,
        deprecated_testing_flag: bool = False,
        device: Device = "cpu",
    ) -> None:
        super().__init__()

        # In parameters
        self.cfg = copy.deepcopy(cfg)
        self.deprecated_testing_flag = deprecated_testing_flag
        self.device = device

        self.env = build_env(self.cfg)
        self.rng = global_seed(self.cfg.seed, self.env)

        ## Submodule Init
        self._build_policy()
        self.intrinsic = build_intrinsic(self.cfg, self.env, self.device)
        self.memory, self.rmv, self.collector = build_data(
            self.cfg, normalise_obs, self.env, self.policy, self.device
        )

        self.trm, self.tm, self.rm, self.ts = build_teleport(
            self.cfg,
            self.rng.build_generator(),
            self.value_container,
            self.env,
            self.device,
        )

        # RMV injection
        if normalise_obs:
            self.policy.transform_pre = self.rmv
            self.rm.transform = self.rmv
            self.tm.transform = self.rmv

        # Logging
        self._build_logging()

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def _build_policy(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def value_container(self) -> HasValue:
        raise NotImplementedError

    @property
    @abstractmethod
    def policy(self) -> Policy:
        raise NotImplementedError

    def _reset(self, obs: NDArray[Any], terminate: bool):
        # Collector actually already resets the policy, so don't need to repeat here
        self.logger.log(
            {
                "reset_terminate": terminate,
                "reset_step": self.tm.episode_step,
                "reset_obs": obs,
            }
        )
        if self.cfg.cats.teleport.enable:
            tid, _, n_obs = self.trm.select(self.collector)
            self.logger.log(
                {
                    "teleport_step": tid,
                    "teleport_obs": n_obs,
                }
            )
        else:
            obs = self._reset_env()
            self.tm.reset(self.collector.env, obs)

    def _reset_env(self):
        """Manual reset of the environment"""
        o, _ = self.collector.env.reset()
        self.collector.obs = o
        return o

    def _build_logging(self):
        self.logger = KittenLogger(
            self.cfg, algorithm="cats", engine=DictEngine, path=self.cfg.log.path
        )
        self.logger.register_providers(
            [
                (self.intrinsic, "intrinsic"),
                (self.collector, "collector"),
                (self.memory, "memory"),
            ]
        )
