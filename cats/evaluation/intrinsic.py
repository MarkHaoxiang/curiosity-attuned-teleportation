from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gymnasium import Env
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch

from kitten.intrinsic.rnd import RandomNetworkDistillation
from kitten.intrinsic.disagreement import Disagreement

from .evaluation import *
from cats.reset import ResetActionWrapper

if TYPE_CHECKING:
    from ..agent.experiment import ExperimentBase


@dataclass
class PseudoTransitions:
    s_0 = None
    a = None
    r = None
    s_1 = None
    d = None
    t = None


def get_state_action_samples(env: Env, samples: int = 10000):
    env.action_space.seed(0)
    env.observation_space.seed(0)
    states = np.array([env.observation_space.sample() for _ in range(10000)])
    actions = np.array([env.action_space.sample() for _ in range(10000)])
    if isinstance(env, ResetActionWrapper):
        actions[:, -1] = 0  # No Resets
    return states, actions


def mcc_step(s, a):
    # Batch step with mountain car transition, but doesn't check action bounds
    # Assumes reasonable action
    POWER = 0.0015
    MIN_POSITION = -1.2
    MAX_POSITION = 0.6
    MAX_SPEED = 0.07

    position = np.expand_dims(s[:, 0], -1)
    velocity = np.expand_dims(s[:, 1], -1)
    force = a
    velocity += force * POWER - 0.0025 * np.cos(3 * position)
    velocity = np.clip(velocity, -MAX_SPEED, MAX_SPEED)
    position += velocity
    position = np.clip(position, MIN_POSITION, MAX_POSITION)
    velocity = velocity * ~((position == MIN_POSITION) & (velocity < 0))

    return np.concatenate((position, velocity), axis=1)


def evaluate_intrinsic(experiment: ExperimentBase, samples: int = 10000):
    if isinstance(experiment.intrinsic, Disagreement):
        return evaluate_disagreement(experiment, samples)
    elif isinstance(experiment.intrinsic, RandomNetworkDistillation):
        return evaluate_rnd(experiment, samples)
    return 0


def evaluate_disagreement(experiment: ExperimentBase, samples: int = 10000) -> float:
    with torch.no_grad():
        assert isinstance(experiment.intrinsic, Disagreement)
        env = experiment.env
        device = experiment.device
        s, a = get_state_action_samples(env, samples)
        s = torch.tensor(s, device=device).to(torch.float32)
        a = torch.tensor(a, device=device).to(torch.float32)
        if experiment.rmv is not None:
            s = experiment.rmv.transform(s)
        batch = PseudoTransitions()
        batch.s_0 = s
        batch.a = a
        return experiment.intrinsic._reward(batch).mean().item()


# Aims to evaluate which trained intrinsic model is the best
def evaluate_rnd(experiment: ExperimentBase, samples: int = 10000) -> float:
    with torch.no_grad():
        env = experiment.env
        device = experiment.device
        s = torch.tensor(get_state_action_samples(env, samples)[0]).to(
            device=device, dtype=torch.float32
        )
        if experiment.rmv is not None:
            s = experiment.rmv.transform(s)
        rnd = experiment.intrinsic
        assert isinstance(rnd, RandomNetworkDistillation)
        return rnd.forward(s).mean().item()


def visualise_rnd(experiment: ExperimentBase, fig: Figure, ax: Axes):
    s, states, x_label, y_label = generate_2d_grid(experiment)
    # RND
    values = experiment.intrinsic(s.to(experiment.device))
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(values.detach().cpu())
    ax.set_title("RND Reward Visualisation")
    ax.scatter(states[:, 0], states[:, 1], c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(m, ax=ax)
