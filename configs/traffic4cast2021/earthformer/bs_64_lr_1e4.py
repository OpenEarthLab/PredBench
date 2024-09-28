_base_ = [
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/traffic4cast2021.py',
    '../../_base_/models/earthformer.py',
    '../../_base_/schedules/schedule_100e.py',
]

batch_size = 16
base_batch_size = 64
base_lr = 1e-4
auto_scale_lr = dict(enable=True, base_batch_size=base_batch_size)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=1.0e-5),
    clip_grad=dict(max_norm=1.0),
)
param_scheduler = [
    dict(type='OneCycleLR',
        eta_max=base_lr, total_steps=None, # automatically compute max iterations: len(train_dataset)*n_epoch/(batch_size*n_gpu)
        final_div_factor=int(1e4), by_epoch=False
    )
]

randomness=dict(seed=0)
custom_hooks = [dict(type='VizVideoHook', data_type='traffic4cast2021')]
visualizer=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='PredBench',
                name='earthformer_traffic4cast2021_bs_64_lr_1e4'
            )
        ),
    ],
)


# dataset settings
train_dataloader = dict(batch_size=batch_size,)
val_dataloader=dict(batch_size=batch_size,)
test_dataloader=dict(batch_size=batch_size,)

# model
transformer = _base_.transformer
transformer.input_shape = (_base_.input_len, _base_.img_H, _base_.img_W, _base_.img_C)
transformer.target_shape = (_base_.output_len,  _base_.img_H, _base_.img_W, _base_.img_C)

model = dict(type='TransformerModel',
    data_processor=_base_.data_processor,
    transformer=transformer,
    loss_fn=dict(type='WeightedLoss'),
)

