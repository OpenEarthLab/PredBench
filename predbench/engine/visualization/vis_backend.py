from typing import Optional, Union
from mmengine.visualization.vis_backend import WandbVisBackend as MMEngine_WandbVisBackend


from predbench.registry import VISBACKENDS


@VISBACKENDS.register_module()
class WandbVisBackend(MMEngine_WandbVisBackend):
    def __init__(self, 
        save_dir: str, 
        init_kwargs: Optional[dict] = None,
        define_metric_cfg: Union[dict, list, None] = None, 
        commit: Optional[bool] = True,
        log_code_name: Optional[str] = None, 
        watch_kwargs: Optional[dict] = None
    ):
        super().__init__(save_dir, init_kwargs, define_metric_cfg, commit, log_code_name, watch_kwargs)

