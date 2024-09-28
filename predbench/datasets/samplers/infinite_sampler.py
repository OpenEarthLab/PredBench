from typing import Optional, Sized
import math
from mmengine.dataset.sampler import InfiniteSampler as MMEngine_InfiniteSampler
from predbench.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class InfiniteSampler(MMEngine_InfiniteSampler):
    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int  = None) -> None:
        super().__init__(dataset, shuffle, seed)
        
    def __len__(self) -> int:
        return math.ceil(self.size / self.world_size)



