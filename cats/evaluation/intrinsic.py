from __future__ import annotations
from typing import TYPE_CHECKING

from gymnasium import Env
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch

from kitten.intrinsic.rnd import RandomNetworkDistillation

from .evaluation import *

if TYPE_CHECKING:
    from ..agent.experiment import ExperimentBase


# Aims to evaluate which trained intrinsic model is the best
def evaluate_rnd(experiment: ExperimentBase, samples: int = 10000) -> float:
    env = experiment.env
    device = experiment.device
    env.observation_space.seed(0)
    s = torch.tensor(
        np.array([env.observation_space.sample() for _ in range(samples)])
    ).to(device=device)
    if experiment.rmv is not None:
        s = experiment.rmv.transform(s)
    rnd = experiment.intrinsic
    assert isinstance(rnd, RandomNetworkDistillation)
    return rnd.forward(s).mean().item()

def visualise_rnd(experiment: ExperimentBase, fig: Figure, ax: Axes):
    s, states, x_label, y_label = generate_2d_grid(experiment)
    # RND
    values = experiment.intrinsic(states.to(experiment.device))
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(values.detach().cpu())
    ax.set_title("RND Reward Visualisation")
    ax.scatter(states[:, 0], states[:, 1], c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(m, ax=ax)