import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpus',type=str, default='0,1,2,3',help='Available gpus separated by comma.')
parser.add_argument('--max_concurrent',type=int, default=24, help='Number of concurrent processes')

parser.add_argument('--num_layers',type=int, default=3)
parser.add_argument('--dataset',type=str, default='mnist', choices=['mnist', 'cifar'])
parser.add_argument('--save_type', type=str, default='none', choices=['none', 'json', 'all'])
FLAGS = parser.parse_args()

#AUG_TYPES = ['noise', 'rotation', 'edge_noise']
AUG_TYPES = ['noise']
SETTINGS  = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
COMMAND = 'python main.py --dataset {dataset} --model {model} --aug {aug} --aug_type {aug_type} --vcp {vcp} --split_for_cifar 2'
stack = []
for aug_type in AUG_TYPES:
    for aug in SETTINGS:
        for model in ['fcn', 'cnn']:
            command = COMMAND.format(
                dataset=FLAGS.dataset,
                model=model,
                aug=aug,
                aug_type=aug_type,
                vcp=0.,
            )
            stack.append(command)

for command in stack:
    subprocess.run(command,shell=True)