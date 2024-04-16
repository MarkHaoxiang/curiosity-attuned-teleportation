# Std
import copy


# Training
import torch
from tqdm import tqdm
from omegaconf import DictConfig

# Kitten
from kitten.experience.util import (
    build_replay_buffer,
    build_transition_from_list,
)
from kitten.experience.collector import GymCollector
from kitten.policy import ColoredNoisePolicy, Policy
from kitten.common import *
from kitten.common.typing import Device
from kitten.common.util import *
from kitten.logging import DictEngine, KittenLogger

# Cats
from .rl import QTOptCats, ResetValueOverloadAux
from .env import *
from .reset import *
from .teleport import *
from .logging import *
from .evaluation import *

from .agent.experiment import ExperimentBase, TeleportationResetModule


class CatsExperiment(ExperimentBase):
    """Experiment baseline"""

    def __init__(
        self,
        cfg: DictConfig,
        deprecated_testing_flag: bool = False,
        device: Device = "cpu",
    ):
        super().__init__(cfg, deprecated_testing_flag, device)
        # Parameters
        self.teleport_cfg = self.cfg.cats.teleport
        self.fixed_reset = self.cfg.cats.fixed_reset
        self.enable_policy_sampling = self.cfg.cats.enable_policy_sampling
        self.death_is_not_the_end = self.cfg.cats.death_not_end
        self.environment_action_noise = self.cfg.cats.env_action_noise
        self.reset_as_an_action = self.cfg.cats.reset_action
        self.logging_path = self.cfg.log.path

        # Init
        self._build_logging()

    def run(self):
        """Default Experiment run"""
        # Initialisation
        self.early_start(self.cfg.train.initial_collection_size)
        self.reset_env()
        self.tm.reset(self.collector.env, self.collector.obs)

        # Main Loop
        for step in tqdm(range(1, self.cfg.train.total_frames + 1)):
            # Collect Data
            s_0_c, a_c, r_c, s_1_c, d_c, t_c = self.collector.collect(n=1)[-1]
            self.rmv.add_tensor_batch(
                torch.tensor(s_1_c, device=self.device).unsqueeze(0)
            )

            # Teleport
            self.tm.update(env=self.collector.env, obs=s_0_c)

            if d_c or t_c:
                self._reset(s_1_c, terminate=d_c)

            # Updates
            mb_size = self.cfg.train.minibatch_size
            batch, aux = self.memory.sample(mb_size)

            # Intrinsic Reward Calculation
            self.intrinsic.update(batch, aux, step=step)
            r_t, r_e, r_i = self.intrinsic.reward(batch)

            # RL Update
            aux = ResetValueOverloadAux(
                weights=aux.weights,
                random=aux.random,
                indices=aux.indices,
                v_1=torch.ones_like(batch.r, device=self.device),
                reset_value_mixin_select=torch.zeros_like(
                    batch.t, device=self.device
                ).bool(),
                reset_value_mixin_enable=False,
            )

            # Update batch
            # Question: Should policy learn the value of resetting?
            if self.death_is_not_the_end or self.reset_as_an_action.enable:
                reset_sample = self.rm.targets()
                c = self.algorithm.critics.sample_networks(2)
                c_1, c_2 = c[0], c[1]
                a_1 = self.algorithm.policy_fn(reset_sample, critic=c_1.target)
                a_2 = self.algorithm.policy_fn(reset_sample, critic=c_2.target)
                target_max_1 = c_1.target.q(reset_sample, a_1).squeeze()
                target_max_2 = c_2.target.q(reset_sample, a_2).squeeze()
                self.reset_value = (
                    torch.minimum(target_max_1, target_max_2).mean().item()
                    - self.reset_as_an_action.penalty
                )
                aux.v_1 = aux.v_1 * self.reset_value
                if self.reset_as_an_action.enable:
                    select = torch.logical_or(batch.t, batch.d)
                elif self.death_is_not_the_end:
                    select = batch.d
                aux.reset_value_mixin_select = select
                aux.reset_value_mixin_enable = True
                # Since we don't consider extrinsic rewards (for now)
                # Manually add the penalty to intrinsic rewards
                if self.reset_as_an_action:
                    r_i = r_i

            if self.death_is_not_the_end:
                batch.d = torch.zeros_like(batch.d, device=self.device).bool()

            batch.r = r_i
            self.algorithm.update(batch, aux, step=step)

            # Evaluation Epoch
            if step % self.cfg.log.frames_per_epoch == 0:
                self.logger.epoch()
                log = {}
                if self.death_is_not_the_end or self.reset_as_an_action:
                    log["reset_value"] = self.reset_value
                # if isinstance(self.intrinsic, RandomNetworkDistillation):
                #     log["evaluate/intrinsic"] = evaluate_rnd(self)
                # log["evaluate/entropy"] = entropy_memory(self.memory.rb)
                self.logger.log(log)

        # Store output
        self.logger.close()

    def early_start(self, n: int):
        """Overrides early start for tighter control

        Args:
            n (int): number of steps
        """
        self.reset_env()
        policy = self.collector.policy
        if not self.reset_as_an_action.enable:
            self.collector.set_policy(Policy(lambda _: self.env.action_space.sample()))
        else:
            # TODO: Edit reset evaluation policy with a small probability of of resets,
            # But teleport right back! To improve initial learning of reset value
            early_start_policy = ResetPolicy(
                self.env, Policy(lambda _: self.env.action_space.sample())
            )
            early_start_policy.enable_evaluation()
            self.collector.set_policy(early_start_policy)
        results = self.collector.early_start(n)
        self.collector.set_policy(policy)
        batch = build_transition_from_list(results, device=self.device)
        self.rmv.add_tensor_batch(batch.s_1)
        batch.s_0, batch.s_1 = self.rmv.transform(batch.s_0), self.rmv.transform(
            batch.s_1
        )
        self.intrinsic.initialise(batch)

    def _reset(self, obs, terminate):
        # Collector actually already resets the policy, so don't need to repeat here
        self.logger.log(
            {
                "reset_terminate": terminate,
                "reset_step": self.tm.episode_step,
                "reset_obs": obs,
            }
        )
        if self.teleport_cfg.enable:
            tid, _, n_obs = self.trm.select(self.collector)
            self.logger.log(
                {
                    "teleport_step": tid,
                    "teleport_obs": n_obs,
                }
            )
        else:
            obs = self.reset_env()
            self.tm.reset(self.collector.env, obs)

    def _build_policy(self):
        self.algorithm = QTOptCats(
            build_critic=lambda: build_critic(self.env, **self.cfg.algorithm.critic).to(
                device=self.device
            ),
            action_space=self.env.action_space,
            obs_space=self.env.observation_space,
            rng=self.rng.build_generator(),
            device=self.device,
            **self.cfg.algorithm,
        )
        self._policy = ColoredNoisePolicy(
            self.algorithm.policy_fn,
            self.env.action_space,
            (
                int(self.env.spec.max_episode_steps)
                if isinstance(self.env.spec.max_episode_steps, int)
                else 1000
            ),
            rng=self.rng.build_generator(),
            device=self.device,
            **self.cfg.noise,
        )
        if self.cfg.cats.reset_action.enable:
            self._policy = ResetPolicy(env=self.env, policy=self._policy)

    @property
    def policy(self):
        return self._policy

    @property
    def value_container(self):
        return self.algorithm

    def _build_logging(self):
        self.logger = KittenLogger(
            self.cfg, algorithm="cats", engine=DictEngine, path=self.logging_path
        )
        self.logger.register_providers(
            [
                (self.algorithm, "train"),
                (self.intrinsic, "intrinsic"),
                (self.collector, "collector"),
                (self.memory, "memory"),
            ]
        )

    def reset_env(self):
        """Manual reset of the environment"""
        if self.enable_policy_sampling:
            self.algorithm.reset_critic()
        o, _ = self.collector.env.reset()
        self.collector.obs = o
        return o
