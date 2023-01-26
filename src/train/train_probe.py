# imports
import transformers
from transformers import set_seed
from datasets import load_dataset
from argparse import ArgumentParser

# hyperparameters
seed = 42

# dataset
dataset_name = 'universal_dependencies'

# tasks
tasks = ['pairwise_distance', 'tree_depth']

# model, tokenizer
model_name = 'xlm-roberta-base'


# directories


# dataset mapping functions
# get pairwise distances for sentences
def tree_distances(batch):
    pass


# process dataset given training task
def process_data(dataset, task):
    pass


def run_probe_training():
    
    # get dataset
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

    # parse cli arguments
    args = argp.parse_args() 

    # training language
    print('Language set to {}'.format(args.lang))
    # training task
    if args.task is None:
        print('specify training task with --task')
        quit()
    if args.task not in tasks:
        print('Task not supported yet. Choose from {}'.format(tasks))
        quit()
    # treebank config
    # support for merging multiple treebanks
    if args.config_list is not None:
        pass
    else:
        print('Using config {}'.format(args.config))
 
    # seed
    set_seed(args.seed)

    # maybe make processed dataset separately?
    # data
    dataset = load_dataset(dataset_name, args.config)

    # process dataset
    process_data(dataset, args.task)
    

    #run_probe_training()