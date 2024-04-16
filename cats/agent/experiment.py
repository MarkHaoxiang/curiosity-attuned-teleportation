import copy
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from kitten.nn import HasValue
from kitten.policy import Policy
from kitten.common.rng import global_seed
from kitten.common.typing import Device

from .env import build_env
from .cats import *


class ExperimentBase(ABC):
    """Contains a set of modules needed to run experiments"""

    def __init__(
        self,
        cfg: DictConfig,
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
            self.cfg, self.env, self.policy, self.device
        )

        self.trm, self.tm, self.rm, self.ts = build_teleport(
            self.cfg,
            self.rng.build_generator(),
            self.value_container,
            self.env,
            self.device,
        )

        # RMV injection
        self.rmv.append(self.policy.fn)
        self.rmv.prepend(self.rm.targets)
        self.rmv.prepend(self.tm.targets)

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
