from predbench.registry import HOOKS

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH

@HOOKS.register_module()
class IterRecordHook(Hook):
    def before_train_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None):
        """
        set model.iter_num to runner._train_loop._iter
        """
        runner.model._iter = runner._train_loop._iter
        