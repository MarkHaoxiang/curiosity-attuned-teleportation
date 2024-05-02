from __future__ import annotations

import math
import logging
from typing import TYPE_CHECKING

import numpy as np
import gymnasium as gym
import torch
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.cm as cm
from sklearn.neighbors import KernelDensity
from gym_continuous_maze.gym_lidar_maze import ContinuousLidarMaze

from kitten.experience.memory import ReplayBuffer
from kitten.experience import Transitions

if TYPE_CHECKING:
    from ..agent.experiment import ExperimentBase


def entropy_memory(memory: ReplayBuffer, ) -> float:
    # Construct a density estimator
    s = Transitions(*memory.sample(len(memory))[0][:5]).s_0.cpu().numpy()
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(s)
    log_likelihoods = kde.score_samples(kde.sample(n_samples=10000))
    return -log_likelihoods.mean()


def visualise_state_targets(
    experiment: ExperimentBase, fig: Figure, ax: Axes, key: str, title: str
):
    env = experiment.env

    try:
        s = np.array(experiment.logger.engine.results[key])
    except:
        logging.log(logging.WARNING, f"Key {key} not found in log")
        return

    s, (x_label, x_low, x_high), (y_label, y_low, y_high) = env_to_2d(env, s)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(s) - 1)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(np.linspace(0, len(s) - 1, len(s)))

    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(x_label)
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel(y_label)
    for i, (x, y) in enumerate(s):
        ax.scatter(x, y, s=2, color=colors[i])
    ax.set_title(title)

    cbar = fig.colorbar(m, ax=ax)
    return fig, ax


def inverse_to_env(env: gym.Env, s):
    match env.spec.id:
        case w if w in ["MountainCarContinuous-v0", "MountainCar-v0", "ContinuousMaze-v0"]:
            return s
        case "ContinuousLidarMaze-v0":
            assert isinstance(env.unwrapped, ContinuousLidarMaze)
            is_tensor = isinstance(s, torch.Tensor)
            if is_tensor:
                s = s.detach().cpu().numpy()
            lidar = []
            for pos in s:
                lidar.append(env.unwrapped.get_lidar_data(pos))
            lidar = np.array(lidar)
            return np.concatenate((s, lidar), axis=1)
        case "Pendulum-v1":
            return np.stack((np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1]), axis=1)
        case _:
            raise ValueError("Environment not yet supported")


def env_to_2d(env: gym.Env, s):
    match env.spec.id:
        case w if w in  ["MountainCarContinuous-v0", "MountainCar-v0"]:
            return (
                s if s is not None else None,
                (
                    "Position",
                    env.observation_space.low[0],
                    env.observation_space.high[0],
                ),
                (
                    "Velocity",
                    env.observation_space.low[1],
                    env.observation_space.high[1],
                ),
            )
        case w if w in ["ContinuousMaze-v0", "ContinuousLidarMaze-v0"]:
            return (
                s[:, :2] if s is not None else None,
                (
                    "X",
                    env.observation_space.low[0],
                    env.observation_space.high[0]
                ),
                (
                    "Y",
                    env.observation_space.low[1],
                    env.observation_space.high[1]
                ),
            )
        case "Pendulum-v1":
            return (
                (
                    np.stack((np.arctan2(s[:, 1], s[:, 0]), s[:, 2]), axis=1)
                    if s is not None
                    else None
                ),
                ("Theta", -math.pi, math.pi),
                (
                    "Angular Velocity",
                    env.observation_space.low[2],
                    env.observation_space.high[2],
                ),
            )
        case _:
            raise ValueError("Environment not yet supported")


def visualise_memory(experiment: ExperimentBase, fig: Figure, ax: Axes, cbar: bool = True):
    """Visualise state space for given environments"""

    env = experiment.env
    memory = experiment.memory.rb
    batch = Transitions(*memory.storage[:5])
    s = batch.s_0.cpu().numpy()

    ax.set_title("State Space Coverage")
    s, (x_label, x_low, x_high), (y_label, y_low, y_high) = env_to_2d(env, s)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel(x_label)
    ax.set_ylim(y_low, y_high)
    ax.set_ylabel(y_label)

    # Colors based on time
    norm = mpl.colors.Normalize(vmin=0, vmax=len(s) - 1)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(np.linspace(0, len(s) - 1, len(s)))
    ax.scatter(s[:, 0], s[:, 1], s=1, c=colors)

    if cbar:
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label("Training Step #", rotation=270, labelpad=15)


def generate_2d_grid(experiment: ExperimentBase):
    _, (x_label, x_low, x_high), (y_label, y_low, y_high) = env_to_2d(
        experiment.env, None
    )
    # Construct Grid
    X = torch.linspace(
        x_low,
        x_high,
        100,
    )
    Y = torch.linspace(
        y_low,
        y_high,
        100,
    )
    grid_X, grid_Y = torch.meshgrid((X, Y))
    states = torch.stack((grid_X.flatten(), grid_Y.flatten())).T
    # Observation Normalisation
    s = torch.tensor(inverse_to_env(experiment.env, states), device=experiment.device, dtype=torch.float32)
    if experiment.rmv is not None:
        s = experiment.rmv.transform(s)
    return s, states, x_label, y_label


def visualise_experiment_value_estimate(
    experiment: ExperimentBase,
    fig: Figure,
    ax: Axes,
    cbar: bool = True
):
    s, states, x_label, y_label = generate_2d_grid(experiment)
    # V
    values = experiment.value_container.value.v(s)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(values.detach().cpu())
    ax.set_title("Value Function Visualisation")
    ax.scatter(states[:, 0], states[:, 1], c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if cbar:
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label("Value", rotation=270, labelpad=15)


def visualise_experiment_policy(
    experiment: ExperimentBase,
    fig: Figure,
    ax: Axes,
    action_axis: int = 0,
):
    # Policy
    s, states, x_label, y_label = generate_2d_grid(experiment)
    actions = experiment.algorithm.policy_fn(s)[:, action_axis]
    if ax is not None:
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(actions.detach().cpu())
        ax.scatter(states[:, 0], states[:, 1], c=colors)
        ax.set_title("Policy Actions")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label("Action", rotation=270, labelpad=15)



def visualise_reset_policy(experiment: ExperimentBase, fig: Figure, ax: Axes):
    # Policy
    s, states, x_label, y_label = generate_2d_grid(experiment)
    actions = experiment.algorithm.policy_fn(s)[:, -1]
    if ax is not None:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = m.to_rgba(actions.detach().cpu())
        ax.scatter(states[:, 0], states[:, 1], c=colors)
        ax.set_title("Reset Actions")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label("Reset (> 0.5)", rotation=270, labelpad=15)


def visualise_experiment_value_reset_estimate(
    experiment: ExperimentBase,
    fig: Figure,
    ax: Axes,
):
    s, states, x_label, y_label = generate_2d_grid(experiment)
    # V
    a = experiment.algorithm.policy_fn(s)
    a[:, -1] = 1  # Reset action
    values = experiment.algorithm.critic.q(s, a)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = m.to_rgba(values.detach().cpu())
    ax.set_title(f"Reset Value {experiment.reset_value}")
    ax.scatter(states[:, 0], states[:, 1], c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.colorbar(m, ax=ax)
