from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch.utils.data import DataLoader
import math
from mmengine.runner.loops import IterBasedTrainLoop as MMengine_IterBasedTrainLoop
from mmengine.dist.utils import get_world_size
from mmengine.runner.utils import calc_dynamic_intervals

from predbench.registry import LOOPS

@LOOPS.register_module()
class InfiniteEpochBasedTrainLoop(MMengine_IterBasedTrainLoop):
    def __init__(self, 
        runner, 
        dataloader, 
        max_epochs, 
        val_begin: int = 1, 
        val_interval: int = 1000, 
        dynamic_intervals: List[Tuple[int, int]] = None
    ) -> None:
        super().__init__(
            runner, dataloader, max_iters=0, val_begin=val_begin, 
            val_interval=0, dynamic_intervals=dynamic_intervals
        )
        world_size = get_world_size()
        self.n_iter_per_epoch = math.ceil(len(self.dataloader.dataset) / (world_size * self.dataloader.batch_size))
        self._max_epochs = max_epochs
        self.val_interval = val_interval * self.n_iter_per_epoch
        self._max_iters = max_epochs * self.n_iter_per_epoch
        
        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)
    
    def run_iter(self, data_batch: Sequence[dict]) -> None:
        super().run_iter(data_batch)
        if self._iter % self.n_iter_per_epoch == 0:
            self._epoch += 1
    