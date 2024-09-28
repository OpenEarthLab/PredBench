
spatio_kernel_enc = 3
spatio_kernel_dec = 3
hid_S = 64
hid_T = 512
N_T = 6
N_S = 4

CNN = dict(
    type = 'MetaVPModel',
    input_shape = (10, 3, 64, 64),  # (T, C, H, W)
    hid_S = hid_S, 
    hid_T = hid_T, 
    N_S = N_S, 
    N_T = N_T, 
    model_type='incepu',
    spatio_kernel_enc=spatio_kernel_enc, 
    spatio_kernel_dec=spatio_kernel_dec, 
    act_inplace=True
)

