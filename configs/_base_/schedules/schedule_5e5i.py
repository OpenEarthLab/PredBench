# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=int(5e5), val_begin=1, val_interval=int(2e4))
val_cfg = dict(type='ValLoop')   
test_cfg = dict(type='TestLoop') 

