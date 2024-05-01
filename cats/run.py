import pickle as pkl
from os.path import join

# Training
import torch
from omegaconf import DictConfig

# Kitten
from kitten.common.util import *
from cats.off_policy_experiment import CatsExperiment
from cats.evaluation import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg: DictConfig, save: bool = True, device=None):
    if device == None:
        device=DEVICE
    experiment = CatsExperiment(
        cfg=cfg,
        device=device,
    )
    experiment.run()
    if save:
        pkl.dump(experiment, open(join(experiment.logger.path, "experiment.pkl"), "wb"))
    return experiment


@hydra.main(version_base=None, config_path="./config", config_name="defaults_offline")
def main(cfg: DictConfig):
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.log.path = path
    run(cfg)


if __name__ == "__main__":
    main()
