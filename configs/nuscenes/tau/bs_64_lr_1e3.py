_base_ = [
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/nuscenes.py',
    '../../_base_/models/tau.py',
    '../../_base_/schedules/schedule_100e.py',
]

batch_size = 32
base_batch_size = 64
base_lr = 1e-3
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



custom_hooks = [dict(type='VizVideoHook', data_type='nuscenes')]
visualizer=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='PredBench',
                name='tau_nuscenes_bs_64_lr_1e3'
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






