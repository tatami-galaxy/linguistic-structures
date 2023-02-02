# imports
import transformers
from transformers import set_seed
from datasets import load_dataset
from argparse import ArgumentParser
import torch.nn as nn
import torch

# hyperparameters
seed = 42

# dataset
dataset_name = 'universal_dependencies'

# tasks
tasks = ['node_distance', 'tree_depth']


# directories


# distance probe class
class DistanceProbe(nn.Module):
    
    def __init__(self, args):
        super(DistanceProbe, self).__init__()
        self.args = args
        args.model_dim = args.model_name.config.hidden_size
        if args.probe_dim is None:  # dxd transform by default
            self.probe_dim = args.model_dim
        self.proj = None  # projecting transformation

    



# depth probe class
class DepthProbe:
    pass


if __name__ == '__main__':

    argp = ArgumentParser()
    # training task
    argp.add_argument('--task', type=str)
    # language
    argp.add_argument('--lang', type=str, default='en')
    # treebank config
    argp.add_argument('--config', type=str, default='en_pud')
    # multiple configs
    argp.add_argument('--config_list', type=list)
    # multiple configs
    argp.add_argument('--seed', type=int, default=seed)
    # model
    argp.add_argument('--model_name', type=str, default='xlm-roberta-base')
    # probe dimension
    argp.add_argument('--probe_dim', type=int, default=None)

    # parse cli arguments
    args = argp.parse_args() 
    

    #run_probe_training()