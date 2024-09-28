from mmengine.optim.scheduler import _ParamScheduler
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler import OneCycleParamScheduler as MMEngine_OneCycleParamScheduler
from mmengine.optim.scheduler.param_scheduler import INF

from predbench.registry import PARAM_SCHEDULERS

from typing import Optional, Sequence, Union

from torch.optim import Optimizer

from mmengine.optim import BaseOptimWrapper

INF = int(1e9)

class OneCycleParamScheduler(MMEngine_OneCycleParamScheduler):
    def __init__(self,
                 optimizer: Union[Optimizer, BaseOptimWrapper],
                 param_name: str,
                 eta_max: float = 0,
                 total_steps: Optional[int] = None,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 div_factor: float = 25.,
                 final_div_factor: float = 1e4,
                 three_phase: bool = False,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):
        self.begin = begin
        self.end = end
        super().__init__(
            optimizer, param_name, eta_max, total_steps, pct_start, anneal_strategy, 
            div_factor, final_div_factor, three_phase, begin, end, last_step, by_epoch, verbose
        )


@PARAM_SCHEDULERS.register_module()
class OneCycleLR(LRSchedulerMixin, OneCycleParamScheduler):
    # https://github1s.com/open-mmlab/mmengine/blob/HEAD/mmengine/optim/scheduler/lr_scheduler.py
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every
    batch. `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in
    one of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation
    of 1cycle, which claims that "unpublished work has shown even better
    results by using only two phases". To mimic the behaviour of the original
    paper instead, set ``three_phase=True``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        eta_max (float or list): Upper parameter value boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by
            providing a value for epochs and steps_per_epoch.
            Defaults to None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Defaults to 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing,
            "linear" for linear annealing.
            Defaults to 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_param = eta_max/div_factor
            Defaults to 25
        final_div_factor (float): Determines the minimum learning rate via
            eta_min = initial_param/final_div_factor
            Defaults to 1e4
        three_phase (bool): If ``True``, use a third phase of the schedule to
            annihilate the learning rate according to 'final_div_factor'
            instead of modifying the second phase (the first two phases will be
            symmetrical about the step indicated by 'pct_start').
        last_step (int): The index of last step. Used for resume without
            state dict. Defaults to -1.
        by_epoch (bool): Whether the scheduled parameters are updated by
            epochs. Defaults to True.
        verbose (bool): Whether to print the value for each update.
            Defaults to False.

    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """# noqa E501





