_base_ = [
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/kth.py',
    '../../_base_/datasets/mau.py'
    '../../_base_/schedules/schedule_100e.py',
]


base_lr = 5e-4
batch_size = 16
base_batch_size = 16
# optimizer 
auto_scale_lr = dict(enable=True, base_batch_size=base_batch_size)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='bfloat16',
    optimizer=dict(type='Adam', lr=base_lr),
    clip_grad=None
)
# learning rate scheduler 
param_scheduler = [
    dict(type='OneCycleLR',
        eta_max=base_lr, total_steps=None, # automatically compute max iterations: len(train_dataset)*n_epoch/(batch_size*n_gpu)
        pct_start=0., div_factor=25, final_div_factor=int(base_lr*int(1e6)), 
        by_epoch=False
    )
]

custom_hooks = [dict(type='VizVideoHook', data_type='kth'), dict(type='IterRecordHook')]
visualizer=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
            init_kwargs=dict(
                project='PredBench',
                name='mau_kth_bs_16_lr_5e4'
            )
        ),
    ],
)


train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)
test_dataloader = dict(batch_size=batch_size)


# reverse scheduled sampling
reverse_scheduled_sampling = False
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = True
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 2e-5

# model
RNN = _base_.RNN
RNN.input_shape = _base_.input_shape

model = dict(type='RNNModel',
    data_processor=_base_.data_processor,
    RNN=RNN,
    loss_fn=dict(type='WeightedLoss'),
    # reverse scheduled sampling
    reverse_scheduled_sampling = reverse_scheduled_sampling,
    r_sampling_step_1 = r_sampling_step_1,
    r_sampling_step_2 = r_sampling_step_2,
    r_exp_alpha = r_exp_alpha,
    # scheduled sampling
    scheduled_sampling = scheduled_sampling,
    sampling_stop_iter = sampling_stop_iter,
    sampling_start_value  = sampling_start_value,
    sampling_changing_rate = sampling_changing_rate,
)


