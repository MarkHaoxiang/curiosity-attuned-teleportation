from omegaconf import DictConfig
from kitten.experience.util import build_replay_buffer
from kitten.policy import Policy
from kitten.experience.collector import GymCollector
from kitten.common.rng import Generator
from kitten.common import util

from cats.teleport import *
from cats.reset import *

from .env import classic_control, minigrid
from .minigrid import build_rnd


class TeleportationResetModule:
    def __init__(
        self, rm: ResetMemory, tm: TeleportMemory, ts: TeleportStrategy
    ) -> None:
        super().__init__()
        self.rm = rm
        self.tm = tm
        self.ts = ts

    def select(self, collector: GymCollector) -> tuple[int, gym.Env, NDArray[Any]]:
        # Get possibilities
        s_t = self.tm.targets()
        s_r = self.rm.targets()
        s = torch.cat((s_t, s_r))
        # Find teleportation target
        tid = self.ts.select(s)
        if tid < len(s_t):
            env, obs = self.tm.select(tid, collector)
        else:
            tid = tid - len(s_t)
            env, obs = self.rm.select(tid, collector)
            self.tm.reset(env, obs)
            tid = 0
        return tid, env, obs


def build_teleport(
    cfg: DictConfig, rng: Generator, value: HasValue, env: gym.Env, device: Device
) -> tuple[TeleportationResetModule, TeleportMemory, ResetMemory, TeleportStrategy]:
    t_cfg = cfg.cats.teleport
    ts_rng, tm_rng = rng.build_generator(), rng.build_generator()
    # Teleport Sampling Strategy
    match t_cfg.type:
        case "e_greedy":
            ts = EpsilonGreedyTeleport(algorithm=value, rng=ts_rng, **t_cfg.kwargs)
        case "boltzmann":
            ts = BoltzmannTeleport(
                algorithm=value,
                rng=ts_rng,
                **t_cfg.kwargs,
            )
        case "ucb":
            ts = UCBTeleport(algorithm=value, **t_cfg.kwargs)
        case _:
            raise ValueError(f"Unknown Teleport Strategy {t_cfg.type}")

    # Teleport Memory
    match t_cfg.memory.type:
        case "fifo":
            tm = FIFOTeleportMemory(
                env=env,
                rng=tm_rng,
                capacity=t_cfg.memory.capacity,
                device=device,
            )
        case "episode":
            tm = LatestEpisodeTeleportMemory(rng=tm_rng, device=device)
        case _:
            raise ValueError(f"Unknown Teleport Memory")

    # Reset Memory
    rm = ResetMemory(
        env=env,
        capacity=1 if cfg.cats.fixed_reset else 128,
        rng=rng.build_generator(),
        device=device,
    )

    # Teleportation Reset Module
    trm = TeleportationResetModule(rm=rm, tm=tm, ts=ts)

    return trm, tm, rm, ts


def build_data(cfg, env: gym.Env, policy: Policy, device: Device):
    memory, rmv = build_replay_buffer(
        env,
        capacity=cfg.train.total_frames + cfg.train.initial_collection_size,
        normalise_observation=True,
        device=device,
    )
    collector = GymCollector(policy, env, memory, device=device)
    return memory, rmv, collector


def build_intrinsic(cfg, env: gym.Env, device: Device = "cpu"):
    if cfg.env.name in classic_control:
        return util.build_intrinsic(env, cfg.intrinsic, device=device)
    elif cfg.env.name in minigrid:
        assert cfg.intrinsic.type == "rnd", "Not yet implemented"
        return build_rnd()
    else:
        raise ValueError("Unknown env name")
