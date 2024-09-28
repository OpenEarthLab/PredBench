_base_ = [
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/sevir.py',
    '../../_base_/models/earthformer.py',
    '../../_base_/schedules/schedule_100e.py',
]

batch_size = 16
base_batch_size = 32
base_lr = 1e-3
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
                name='earthformer_sevir_bs_32_lr_1e3'
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
transformer.base_units = 128
transformer.num_global_vectors = 8
transformer.use_global_self_attn = True
transformer.separate_global_qkv = True
transformer.initial_downsample_type = 'stack_conv'
transformer.initial_downsample_activation = 'leaky'
transformer.initial_downsample_scale = 1
transformer.initial_downsample_stack_conv_num_layers = 3
transformer.initial_downsample_stack_conv_dim_list = [16, 64, 128]
transformer.initial_downsample_stack_conv_downscale_list = [3, 2, 2]
transformer.initial_downsample_stack_conv_num_conv_list = [2, 2, 2]


model = dict(type='TransformerModel',
    data_processor=_base_.data_processor,
    transformer=transformer,
    loss_fn=dict(type='WeightedLoss'),
)

