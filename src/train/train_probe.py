# imports
from os.path import dirname, abspath
import transformers
from transformers import set_seed
from datasets import load_from_disk
from argparse import ArgumentParser
import torch.nn as nn
import torch


# directories

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'linguistic-structures':
    root = dirname(root)
    
processed_data_dir = root+'/data/processed/UD/'

# tasks
tasks = ['node_distance', 'tree_depth']


# directories


# distance probe class
class DistanceProbe(nn.Module):
    
    def __init__(self, args):
        super(DistanceProbe, self).__init__()
        self.args = args
        self.model_dim = args.model_name.config.hidden_size
        if args.probe_dim is None:  # dxd transform by default
            self.probe_rank = self.model_dim
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))  # projecting transformation # device?
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, batch):
        transformed = torch.matmul(batch, proj) # b,s,r
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2) # b, s, 1, r
        transformed = transformed.expand(-1, -1, seqlen, -1) # b, s, s, r
        transposed = transformed.transpose(1,2) # b, s, s, r
        diffs = transformed - transposed # b, s, s, r
        squared_diffs = diffs.pow(2) # b, s, s, r
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances

    



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
    argp.add_argument('--seed', type=int, default=42)
    # model
    argp.add_argument('--model_name', type=str, default='xlm-roberta-base')
    # probe dimension
    argp.add_argument('--probe_rank', type=int, default=None)

    # parse cli arguments
    args = argp.parse_args() 

    # seed
    set_seed(args.seed)

    # task
    if args.task is None:
        print('specify training task with --task, from {}'.format(tasks))
        quit()
    task = args.task

    # 'idx', 'text', 'tokens', 'lemmas', 'upos', 'xpos', 'feats', 
    # 'head', 'deprel', 'deps', 'misc', 'input_ids', 'attention_mask', 'true_dist'
    dataset = load_from_disk(processed_data_dir+args.config+'_'+task)
    print(dataset['test'][0]['input_ids'])
    print(len(dataset['test'][0]['input_ids']))
    print(len(dataset['test'][0]['attention_mask']))
    print(len(dataset['test'][0]['true_dist']))

    #run_probe_training()