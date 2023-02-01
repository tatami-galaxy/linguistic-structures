# imports
import transformers
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

# hyperparameters
seed = 42
max_length = 256

# dataset
dataset_name = 'universal_dependencies'

# tasks
tasks = ['node_distance', 'tree_depth']


# directories
processed_data_dir = '/users/ujan/linguistic-structures/data/processed/UD/'

# UD class

class UD:

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

    # dataset mapping functions
    # get pairwise distances for sentences
    def tree_distances(self, heads):
        # compute adj matrix from heads
        # floyd warshall
        dists = []
        batch_size = len(heads)
        for b in range(batch_size):
            head_list = heads[b]
            seq_len = len(head_list)
            graph = np.zeros((seq_len, seq_len))
            # populate adjacency matrix
            for s in range(seq_len):
                if head_list[s] != 0:  # 0 is root
                    graph[s, int(head_list[s])-1] = 1  # token id starts from 1 since 0 means root
            # all pair shortest path 
            dist_matrix = floyd_warshall(csgraph=graph, directed=False, unweighted=True)
            dists.append(dist_matrix)
        return dists


    def distance_map(self, batch):

        ids = batch['idx']
        tokens = batch['tokens']
        text = batch['text']
        heads = batch['head']

        # compute true distances from heads
        # tokenized batch -> model -> average subword embeddings -> probe -> distances

        model_inputs = self.tokenizer(
            tokens, truncation=True,
            max_length=self.args.max_length,
            is_split_into_words=True)

        # true distance matrix
        true_dists = self.tree_distances(heads) 
        model_inputs['true_dist'] = true_dists

        # model_inputs.word_ids(i) maps tokenized tokens
        # of ith sample in batch to input tokens
        # after removing special characters

        return model_inputs



    # process dataset given training task
    def process_data(self, dataset_name, args):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dataset = load_dataset(dataset_name, args.config)

        # filter examples with None in head
        dataset = dataset.filter(lambda example: all(h.isdigit() for h in example['head']))

        # number of tokens in UD <= number of tokens in tokenized text
        # need to map tokenized tokens to UD tokens
        if args.task == 'node_distance':
            # inputs_ids, attention_mask, true_dists
            dataset = dataset.map(self.distance_map, batched=True)
            return dataset





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
        print('specify training task with --task, from {}'.format(tasks))
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
    dataset = ud.process_data(dataset_name, args)
    dataset.save_to_disk(processed_data_dir+args.config+'_'+args.task)


    
