import torch
from torch import Tensor

# Taken from 
    # minigrid.core.constants
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

def one_hot_encoding(x: Tensor):
    # (batch, channel, x, y)
    # TODO: Lava
    assert len(x.shape) == 4
    assert x.shape[-3:] == (3,19,19)
    idx = x[..., 0, :, :]
        # We want to extract "Empty, Wall, Agent, Goal
    image = torch.stack([
        idx == i for i in [1,2,8,10]
    ], dim=1).to(torch.float32) 
        # Agent Direction
    dir = ((idx==10) * x[..., 2, :, :]).sum(dim=(-1,-2))
    dir_vec = torch.stack([
        dir==i for i in [0,1,2,3]
    ], dim=1).to(torch.float32)

    return image, dir_vec
