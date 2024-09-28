# training schedule
train_cfg = dict(type='IterBasedTrainLoop', max_iters=int(2e6), val_begin=1, val_interval=int(2e4))
val_cfg = dict(type='ValLoop')   
test_cfg = dict(type='TestLoop')  

