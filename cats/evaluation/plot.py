import matplotlib.pyplot as plt

from .evaluation import *
from .intrinsic import *

def visualise_classic_control_results(experiment):
    log = experiment.logger.engine.results

    # Set up visualisation
    N_ROWS, N_COL = 4, 4
    fig, axs = plt.subplots(N_ROWS, N_COL)
    fig.set_size_inches(N_COL * 6, N_ROWS * 5)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    row = 0
    # General Status
    visualise_memory(experiment, fig, axs[row][0])
    visualise_experiment_value_estimate(experiment, fig, axs[row][1])
    if hasattr(experiment, "algorithm"):
        visualise_experiment_policy(experiment, fig, axs[row][2])
    else:
        axs[row][2].axis("off")
    ax = axs[row][3]
    ax.plot(log["collector/frame"])
    ax.set_title("Frames Collected")
    ax.set_ylabel("Frames")
    ax.set_xlabel("Epoch")

    # Training Progress
    row += 1

    ax = axs[row][0]
    ax.set_ylabel("Loss")
    if "train/critic_loss" in log:
        ax.plot(log["train/critic_loss"])
        ax.set_title("Critic Loss")
        ax.set_xlabel("Epoch")
    elif "train/value_loss" in log:
        ax.plot(log["train/value_loss"])
        ax.set_title("Value Loss")
        ax.set_xlabel("Epoch")
    else:
        ax.axis("off")

    ax = axs[row][1]
    ax.plot(log["intrinsic/loss"])
    ax.set_yscale("log")
    ax.set_title("Intrinsic Loss (Train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax = axs[row][2]
    if "evaluate/entropy" in log:
        ax.plot(log["evaluate/entropy"])
        ax.set_title("Exploration Entropy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dataset Entropy")
    else:
        ax.axis("off")

    ax = axs[row][3]
    if "evaluate/intrinsic" in log:
        ax.plot(log["evaluate/intrinsic"])
        ax.set_title("Intrinsic Loss (Evaluate)")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.set_ylabel("Loss")
    else:
        ax.axis("off")

    # Resets
    row += 1
    visualise_state_targets(
        experiment, fig, axs[row][0], key="reset_obs", title="Reset Observations"
    )

    ax = axs[row][1]
    if "reset_terminate" in log:
        colors = ["red" if x else "blue" for x in log["reset_terminate"]]
        x = range(len(log["reset_step"]))
        ax.scatter(x, log["reset_step"], color=colors, s=5)
        ax.set_title("Reset Step in Episode")
        ax.set_xlabel("Occurence")
        ax.set_ylabel("Step From Episode Reset")
        if experiment.cfg.cats.teleport.enable:
            ax.set_title("Reset / Teleport Step in Episode")
            ax.scatter(range(len(log["teleport_step"])), log["teleport_step"], s=5)
    else:
        ax.axis("off")
        

    if experiment.cfg.cats.reset_action.enable:
        visualise_reset_policy(experiment, fig, axs[row][2])
        visualise_experiment_value_reset_estimate(experiment, fig, axs[row][3])
    else:
        axs[row][2].axis("off")
        axs[row][3].axis("off")

    # Teleports
    row += 1
    if experiment.cfg.cats.teleport.enable:
        visualise_state_targets(
            experiment,
            fig,
            axs[row][0],
            key="teleport_obs",
            title="Teleport Observations",
        )
    else:
        axs[row][0].axis("off")

    # Intrinsic
    if isinstance(experiment.intrinsic, RandomNetworkDistillation):
        visualise_rnd(experiment, fig, ax=axs[row][1])

    # Final Values
    print(f"Entropy: {entropy_memory(experiment.memory.rb)}")
    if isinstance(experiment.intrinsic, RandomNetworkDistillation):
        print(f"RND: {evaluate_rnd(experiment)}")
    elif isinstance(experiment.intrinsic, Disagreement):
        print(f"Disagreement {evaluate_disagreement(experiment)}")

    axs[row][2].axis("off")
    axs[row][3].axis("off")

rng = np.random.default_rng(0)
seeds = [rng.integers(2**32-1) for _ in range(10)]