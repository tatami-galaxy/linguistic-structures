# imports
import transformers
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser

# hyperparameters
seed = 42
max_length = 256

# dataset
dataset_name = 'universal_dependencies'

# tasks
tasks = ['pairwise_distance', 'tree_depth']


# directories

# UD class

class UD:

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

    # dataset mapping functions
    # get pairwise distances for sentences
    def tree_distances(self, batch):
        pass


    def distance_map(self, batch):

        ids = batch['idx']
        tokens = batch['tokens']
        text = batch['text']
        heads = batch['head']

        tokenized_batch = self.tokenizer(
            tokens, truncation=True,
            max_length=self.args.max_length,
            is_split_into_words=True)

        # word_ids(i) maps tokenized tokens to input tokens
        # after removing special characters

        # compute adj matrix




# process dataset given training task
def process_data(self, dataset_name, config, tokenizer, task):
    dataset = load_dataset(dataset_name, config)
    # number of tokens in UD <= number of tokens in tokenized text
    # need to map tokenized tokens to UD tokens
    print(dataset)
    quit()




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
    argp.add_argument('--model', type=str, default='xlm-roberta-base')
    # max length
    argp.add_argument('--max_length', type=int, default=max_length)

    # parse cli arguments
    args = argp.parse_args() 

    # seed
    set_seed(args.seed)

    # data language
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

    # UD processing class
    ud = UD(args)

    # get data and process 
    _  = ud.process_data(dataset_name, args.config, args.task)
    
