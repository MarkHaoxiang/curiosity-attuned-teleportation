from copy import deepcopy
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import torch

from kitten.rl.common import generate_minibatches, td_lambda
from kitten.experience import AuxiliaryMemoryData, Transitions
from kitten.experience.util import build_transition_from_list
from kitten.nn import Value, Ensemble

from cats.agent.experiment import ExperimentBase


class OnlineExperiment(ExperimentBase, ABC):

    def __init__(
        self,
        cfg: DictConfig,
        deprecated_testing_flag: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__(cfg,
                         normalise_obs=True,
                         deprecated_testing_flag=deprecated_testing_flag,
                         device=device)
        self._gamma: float = cfg.algorithm.gamma
        self._lmbda: float = cfg.algorithm.lmbda
        self._mb_size: int = cfg.algorithm.minibatch_size
        self._n_update_epochs: int = cfg.algorithm.n_update_epochs
    def _build_policy(self) -> None:
        self._ucb_enabled = self.cfg.cats.teleport.enable and self.cfg.cats.teleport.type == "ucb"
        if self._ucb_enabled:
            self._value = Ensemble(self._build_value, n=5, rng=self.rng.build_generator())
        else:
            self._value = self._build_value()
        self.optim_v = torch.optim.Adam(params=self._value.parameters()) 
    
    @property
    def value_container(self):
        return self

    def mu_var(self, s):
        assert self._ucb_enabled
        with torch.no_grad():
            return self._value.mu_var(s)

    @property
    def value(self):
        if isinstance(self._value, Ensemble):
            return self._value.sample_network()
        else:
            return self._value

    @abstractmethod
    def _build_value(self) -> Value:
        raise NotImplementedError

    def value_update(self, batch: Transitions) -> float:
        if not self._ucb_enabled:
            value = [self._value]
        else:
            value = self._value

        # Different Bootstrapped Targets
        value_targets = []
        for i, v in enumerate(value):
            value_targets.append(td_lambda(batch, self._lmbda, self._gamma, v))

        value_targets = torch.stack(value_targets)
        bit_mask = torch.rand(value_targets.shape, device=self.device) < 0.5
        value_targets = value_targets * bit_mask

        total_value_loss = 0
        for _ in range(self._n_update_epochs):
            for i, _ in generate_minibatches(value_targets[0], mb_size=self._mb_size, rng=self.rng):
                self.optim_v.zero_grad()
                value_loss = 0
                for j, v in enumerate(value):
                    pred_value = bit_mask[j][i] * v(batch.s_0[i]).squeeze()
                    value_loss += ((value_targets[j][i] - pred_value) ** 2).mean()
                total_value_loss += value_loss.item()
                value_loss.backward()
                self.optim_v.step()
        return total_value_loss
    
    def policy_update(self, batch: Transitions, step: int):
        pass

    def run(self):
        step, steps = 0, self.cfg.train.total_frames

        if self.rmv is not None:
            batch = self.collector.early_start(n=1000, append_memory=False)
            batch = build_transition_from_list(batch, device=self.device)
            self.rmv.add_tensor_batch(batch.s_0)
        self.collector.obs, _ = self.collector.env.reset()

        self.tm.reset(self.collector.env, self.collector.obs)
        while step < steps:
            # Collect batch
            batch_ = []
            goal_step = step + self.cfg.algorithm.collection_batch
            while step < goal_step:
                step += 1
                data = self.collector.collect(n=1)[-1]
                batch_.append(data)
                self.tm.update(self.collector.env, obs=data[0])
                if data[-2] or data[-1]: # Terminated or truncated
                    break
            batch = build_transition_from_list(batch_, device=self.device)
            
            if self.rmv is not None:
                batch.s_0 = self.rmv.transform(batch.s_0)
                batch.s_1 = self.rmv.transform(batch.s_1)

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
            self.policy_update(batch, step)

            # Always reset? Or only reset on truncation?
            self._reset(batch_[-1][0], batch_[-1][4])
            self.logger.log({"train/value_loss": total_value_loss})
            self.logger.epoch()

from kitten.policy import Policy
from kitten.nn import ClassicalDiscreteStochasticActor, ClassicalValue
from kitten.rl.advantage import GeneralisedAdvantageEstimator
from kitten.rl.ppo import ProximalPolicyOptimisation
class PPOExperiment(OnlineExperiment):
    def _build_policy(self) -> None:
        super()._build_policy()
        gamma, lmbda = self.cfg.algorithm.gamma, self.cfg.algorithm.lmbda
        self._actor = ClassicalDiscreteStochasticActor(self.env, self.rng.build_generator()).to(self.device)
        gae = GeneralisedAdvantageEstimator(self.value, lmbda, gamma)
        self._ppo = ProximalPolicyOptimisation(
            actor=self._actor,
            advantage_estimation=gae,
            rng=self.rng.build_generator()
        )
        self._policy = Policy(fn=self._ppo.policy_fn, device=self.device)
    
    @property
    def policy(self):
        return self._policy

    def _build_value(self) -> Value:
        return ClassicalValue(self.env).to(self.device) 
    
    def policy_update(self, batch: Transitions, step: int):
        super().policy_update(batch, step)
        self._ppo.update(batch, aux=None, step=step)
