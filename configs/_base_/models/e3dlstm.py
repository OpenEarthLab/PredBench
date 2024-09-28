num_hidden = [64,64,64,64]
filter_size = [2, 5, 5]
stride = 1
patch_size = 4
layer_norm = 0

RNN=dict(
    type='E3DLSTM',
    input_shape=(10, 3, 63, 64),    # (T, C, H, W)
    num_layers=len(num_hidden), 
    num_hidden=num_hidden, 
    patch_size=patch_size, 
    filter_size=filter_size, 
    stride=stride, 
    layer_norm=layer_norm, 
)