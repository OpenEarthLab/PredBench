num_hidden = []
patch_size = 1
sr_size = 2


RNN=dict(
    type='PhyDNet',
    input_shape=(10, 3, 63, 64),    # (T, C, H, W)
    num_layers=len(num_hidden), 
    num_hidden=num_hidden, 
    patch_size=patch_size,
    sr_size=sr_size, 
)