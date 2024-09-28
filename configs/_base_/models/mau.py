num_hidden = [64,64,64,64]
filter_size = 5
stride = 1
patch_size = 1
sr_size = 2
tau = 5
cell_mode = 'normal'
model_mode = 'normal'


RNN=dict(
    type='MAU',
    input_shape=(10, 3, 63, 64),    # (T, C, H, W)
    num_layers=len(num_hidden), 
    num_hidden=num_hidden, 
    patch_size=patch_size, 
    filter_size=filter_size, 
    stride=stride, 
    sr_size=sr_size,
    tau=tau, 
    cell_mode=cell_mode,
    model_mode=model_mode,
)