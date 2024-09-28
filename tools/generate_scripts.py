import os
import os.path as osp

def generate_script_with_config():
    os.makedirs('tools/scripts', exist_ok=True)
    dataset_dirs = os.listdir('configs')
    dataset_dirs.remove('_base_')
    for dataset in dataset_dirs:
        os.makedirs(f'tools/scripts/{dataset}', exist_ok=True)
        dataset_dir = osp.join('configs', dataset)
        model_dirs = os.listdir(dataset_dir)
        for model in model_dirs:
            os.makedirs(f'tools/scripts/{dataset}/{model}', exist_ok=True)
            model_dir = osp.join(dataset_dir, model)
            configs = os.listdir(model_dir)
            for config in configs:
                if config.startswith('bs_') and config.endswith('.py'):
                    config_name = config.split('.')[0]
                    with open(f'tools/scripts/{dataset}/{model}/{config_name}.sh', 'w') as f:
                        f.write('#!/usr/bin/env bash\n')
                        f.write('export MASTER_PORT=$((12000 + $RANDOM%20000))\n')
                        f.write('export PYTHONPATH="$\{PYTHONPATH\}:$(pwd)"\n')
                        f.write('\n')
                        f.write('NUM_NODE=${1:-1}\n')
                        f.write('NUM_GPU=${2:-4}\n')
                        f.write('\n')
                        f.write('python -m torch.distributed.run --nnodes=${NUM_NODE} --nproc_per_node=${NUM_GPU} --master_port=${MASTER_PORT} tools/train.py \\\n\t')
                        f.write(f'--config {osp.join(model_dir, config)} \\\n\t')
                        f.write(f'--work-dir work_dirs/{dataset}/{model}/{config_name} \\\n\t')
                        f.write('--launcher pytorch\n')    
                        f.write('\n')
                        f.write('python tools/test.py \\\n\t')
                        f.write(f'--work-dir work_dirs/{dataset}/{model}/{config_name} \\\n\t')                        
                        f.write('--test-best --metric-only\n')


if __name__ == '__main__':
    generate_script_with_config()
