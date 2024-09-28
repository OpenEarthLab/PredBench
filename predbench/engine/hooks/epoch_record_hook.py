from predbench.registry import HOOKS

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH

@HOOKS.register_module()
class EpochRecordHook(Hook):
    def before_train_iter(self, runner, batch_idx: int, data_batch: DATA_BATCH = None):
        """
        set model._epoch to runner._train_loop._epoch
        """
        runner.model._epoch = runner._train_loop._epoch
        