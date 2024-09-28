import torch
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(description='Change model state dict name')
parser.add_argument(
    'ckpt_path', help='the path of the original model',
)
parser.add_argument(
    'new_ckpt_path', help='save the new model in this path'
)

def remove_module_in_state_dict(ckpt_path, new_ckpt_path):
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(list(ckpt.keys()))
    # ['meta', 'state_dict', 'message_hub', 'optimizer', 'param_schedulers']
    # print(sd['meta'])
    # print(sd['message_hub'])
    # print(sd['param_schedulers'])
    
    sd = ckpt['state_dict']
    new_sd = OrderedDict()
    for k, v in sd.items():
        if 'module' in k:
            new_k = k.replace('module.', '')
        else: 
            new_k = k
        new_sd[new_k] = v
        
    ckpt['state_dict'] = new_sd
    
    torch.save(ckpt, new_ckpt_path)
        
    key_zip = zip(new_sd.keys(), sd.keys())
    for keys in key_zip:
        print(keys)
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    ckpt_path = args.ckpt_path
    new_ckpt_path = args.new_ckpt_path
    remove_module_in_state_dict(ckpt_path, new_ckpt_path)