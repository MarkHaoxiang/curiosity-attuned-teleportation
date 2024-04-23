from abc import ABC, abstractmethod
from omegaconf import DictConfig

import torch

from kitten.rl.common import generate_minibatches, td_lambda
from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.experience.util import build_transition_from_list
from kitten.nn import Value

from cats.agent.experiment import ExperimentBase


class OnlineExperiment(ExperimentBase, ABC):

    def __init__(
        self,
        cfg: DictConfig,
        deprecated_testing_flag: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(cfg,
                         normalise_obs=False,
                         deprecated_testing_flag=deprecated_testing_flag,
                         device=device)
        self._gamma: float = cfg.algorithm.gamma
        self._lmbda: float = cfg.algorithm.lmbda
        self._mb_size: int = cfg.algorithm.minibatch_size
        self._n_update_epochs: int = cfg.algorithm.n_update_epochs

    def _build_policy(self) -> None:
        self.value = self._build_value()
        self.optim_v = torch.optim.Adam(params=self.value.parameters()) 
    
    @property
    def value_container(self):
        return self

    @abstractmethod
    def _build_value(self) -> Value:
        raise NotImplementedError

    def value_update(self, batch: Transitions) -> float:
        value_targets = td_lambda(batch, self._lmbda, self._gamma, self.value)
        total_value_loss = 0
        for _ in range(self._n_update_epochs):
            for i, mb in generate_minibatches(value_targets, mb_size=self._mb_size, rng=self.rng):
                self.optim_v.zero_grad()
                value_loss = ((mb - self.value.v(batch.s_0[i])) ** 2).mean()
                total_value_loss += value_loss.item()
                value_loss.backward()
                self.optim_v.step()
        return total_value_loss
    
    def policy_update(self, batch: Transitions):
        pass

    def run(self):
        if self.cfg.cats.teleport.enable:
            self.tm.reset(self.collector.env, self.collector.obs)
        else:
            self._reset_env()
        step, steps = 0, self.cfg.train.total_frames
        while step < steps:
            # Collect batch
            batch_ = []
            goal_step = step + self.cfg.algorithm.collection_batch
            while step < goal_step:
                step += 1
                data = self.collector.collect(n=1)[-1]
                batch_.append(data)
                if self.cfg.cats.teleport.enable:
                    self.tm.update(self.collector.env, obs=data[0])
                if data[-2] or data[-1]: # Terminated or truncated
                    break
            batch = build_transition_from_list(batch_, device=self.device)

            # Intrinsic Update
            for _ in range(self._n_update_epochs):
                for i, mb in generate_minibatches(batch, mb_size=self._mb_size, rng=self.rng):
                    self.intrinsic.update(mb, aux=AuxiliaryMemoryData.placeholder(mb), step=step) 
            # Override rewards
            _, _, r_i = self.intrinsic.reward(batch)
            batch.r = r_i

            # Value Update
            total_value_loss = self.value_update(batch)

            # Policy Update (If needed)
            self.policy_update(batch)

            # Always reset? Or only reset on truncation?
            self._reset(batch_[-1][0], batch_[-1][4])
            self.logger.log({"train/value_loss": total_value_loss})
            self.logger.epoch()