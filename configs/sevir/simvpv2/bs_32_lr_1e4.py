_base_ = [
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/sevir.py',
    '../../_base_/models/simvpv2.py',
    '../../_base_/schedules/schedule_100e.py',
]

batch_size = 32
base_batch_size = 32
base_lr = 1e-4
auto_scale_lr = dict(enable=True, base_batch_size=base_batch_size)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='Adam', lr=base_lr),
    clip_grad=None
)
param_scheduler = [
    dict(type='OneCycleLR',
        eta_max=base_lr, total_steps=None, # automatically compute max iterations: len(train_dataset)*n_epoch/(batch_size*n_gpu)
        final_div_factor=int(base_lr*int(1e6)), by_epoch=False
    )
]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best=['error/mse', 'weather/csi_avg'], rule=['less', 'greater']),
)
custom_hooks = [dict(type='VizVideoHook', data_type='sevir')]
visualizer=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='PredBench',
                name='simvpv2_sevir_bs_32_lr_1e4'
            )
        ),
    ],
)



# dataset settings
train_dataloader = dict(batch_size=batch_size,)
val_dataloader=dict(batch_size=batch_size,)
test_dataloader=dict(batch_size=batch_size,)



# model
CNN = _base_.CNN 
CNN.input_shape = _base_.input_shape

model = dict(type='CNNModel',
    data_processor=_base_.data_processor,
    CNN=CNN,
    loss_fn=dict(type='WeightedLoss')
)






