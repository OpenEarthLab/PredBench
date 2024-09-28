_base_ = [
    './human_scheduler.py',
]

batch_size = 32
base_lr = 1e-4
base_batch_size = 64
auto_scale_lr = dict(enable=True, base_batch_size=base_batch_size)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='Adam', 
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.0, amsgrad=False, eps=0.00000001
    ),
    clip_grad=dict(max_norm=1.0),   # torch.nn.utils.clip_grad_norm_
)
param_scheduler = [
    dict(type='LinearLR',
        start_factor=float(1/5000), begin=0, end=5000, by_epoch=False, 
    )
]


visualizer=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='PredBench',
                name='mcvd_human_bs_64_lr_1e4'
            )
        ),
    ],
)


# dataset settings
train_dataloader = dict(batch_size=batch_size,)
val_dataloader=dict(batch_size=batch_size,)
test_dataloader=dict(batch_size=batch_size,)

input_len = _base_.input_len
output_len = _base_.output_len
pred_len = _base_.pred_len
img_norm_cfg = _base_.img_norm_cfg
img_H = _base_.img_H
img_W = _base_.img_W
img_C = _base_.img_C

# model
score_net = _base_.score_net
score_net.input_output_shape = (_base_.input_len, _base_.output_len, _base_.img_C, _base_.img_H, _base_.img_W)

model = dict(type='DiffusionModel',
    data_processor=_base_.data_processor,
    sampler=_base_.sampler,
    score_net=score_net,
    loss_fn = dict(type='AnnealDSMScoreDstimation',
        loss_type='mse', gamma=_base_.gamma, all_frames=False, 
    ), 
    prob_cond_mask=0.0,
    gamma=False
)

