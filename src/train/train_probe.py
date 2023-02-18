# imports
import os
from os.path import dirname, abspath
from typing import Any, Dict, List, Optional
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from transformers import set_seed
from datasets import load_dataset, load_from_disk
import argparse
from argparse import ArgumentParser
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from scipy.sparse.csgraph import floyd_warshall
from dataclasses import dataclass
import math

# directories

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'linguistic-structures':
    root = dirname(root)
    

# tasks
tasks = ['node_distance', 'tree_depth']

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: XLMRobertaTokenizerFast
    padding: bool = True
    return_tensors: str = "pt"

    # custom padding function
    ##### fix #####
    def custom_pad(self, key, features, pad_token_id):
        lens = [len(x[key]) for x in features] # true lens
        input_list = []
        mask_list = []
        max_len = max(lens)
        for i in range(len(lens)): # len(lens) = batch_size 
            inputs = torch.tensor(features[i][key])
            mask = torch.ones(inputs.shape) # 1 for true positions

            if len(inputs.shape) == 2:  # distance matrix 
                inputs = nn.functional.pad(inputs, (0, max_len-lens[i], 0, max_len-lens[i]), value=pad_token_id)
                mask = nn.functional.pad(mask, (0, max_len-lens[i], 0, max_len-lens[i])) # pad 0 by default
            elif len(inputs.shape) == 1: # array
                inputs = nn.functional.pad(inputs, (0, max_len-lens[i]), value=pad_token_id)
                mask = nn.functional.pad(mask, (0, max_len-lens[i])) # pad 0 by default
            input_list.append(inputs)
            mask_list.append(mask)

        labels = torch.stack(input_list)
        label_mask = torch.stack(mask_list)

        return labels, label_mask, torch.tensor(lens)


    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"]} for feature in features]
        label_features = [{"true_dist": feature["true_dist"]} for feature in features]

        word_ids = [{"word_ids": feature["word_ids"]} for feature in features]

        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )

        # pad true_dist

        # label_features -> list (len = batch_size) of dicts
        # dict -> 'true_dist' : list (len = seq_len x seq_len) of distances (float)
        # tokenizer.pad_token_id = 1
        # -100 for None (special tokens) and padding
        # 1 for true and 0 for pad in mask
        batch["labels"], batch["label_mask"], batch["lens"] = self.custom_pad('true_dist', label_features, -100)
        batch["word_ids"], _, _ = self.custom_pad('word_ids', word_ids, -100)

        return batch


# UD class

class UD:

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

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
            # all pair shortest paths
            dist_matrix = floyd_warshall(csgraph=graph, directed=False, unweighted=True)

            # each matrix will be seq_len x seq_len
            # seq_len varies
            # need to pad into a batch
            #dist_matrix_f = torch.tensor(dist_matrix)

            dists.append(dist_matrix)

        return dists


    def distance_map(self, batch):

        #ids = batch['idx']
        tokens = batch['tokens']
        #text = batch['text']
        heads = batch['head']

        # compute true distances from heads
        # tokenized batch -> model -> average subword embeddings -> probe -> distances

        model_inputs = self.tokenizer(
            tokens, truncation=True,
            max_length=self.args.max_length,
            is_split_into_words=True)
        
        # store word ids
        # -100 for None. maybe something else?
        
        word_ids = [[-100 if x is None else x for x in model_inputs.word_ids(i)] for i in range(len(model_inputs['input_ids']))]
        model_inputs['word_ids'] = word_ids

        # true distance matrix
        true_dists = self.tree_distances(heads) 
        model_inputs['true_dist'] = true_dists

        # model_inputs.word_ids(i) maps tokenized tokens
        # of ith sample in batch to input tokens
        # after removing special characters
        # len(set(model_inputs.word_ids(i))) after removing None 
        # should be equal to len(tokens[i])
        # average embeddings for model_inputs mapping to the same token

        return model_inputs


    # process dataset given training task
    def process_data(self, args):
        dataset = load_dataset(args.dataset_name, args.config)

        # filter examples with None in head
        dataset = dataset.filter(lambda example: all(h.isdigit() for h in example['head']))

        # number of tokens in UD <= number of tokens in tokenized text
        # need to map tokenized tokens to UD tokens
        if args.task == 'node_distance':
            # inputs_ids, attention_mask, true_dists
            dataset = dataset.map(self.distance_map, batched=True, remove_columns=dataset['test'].column_names)
            return dataset



# distance probe class
class DistanceProbe(nn.Module):
    
    def __init__(self, model_dim, probe_rank):
        super(DistanceProbe, self).__init__()
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        if args.probe_rank is None:  # dxd transform by default
            self.probe_rank = self.model_dim
        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))  # projecting transformation # device?
        nn.init.uniform_(self.proj, -0.05, 0.05)


    def del_embeds(self, hidden, ids):
        new_hidden = hidden[:ids[0]]
        prev_id = ids[0]
        for i in range(1, len(ids)):
            new_hidden = torch.cat((new_hidden, hidden[prev_id+1:ids[i]]))
            prev_id = ids[i]
        return new_hidden


    def avg_embed(self, hidden, word_id):
        # get unique counts
        _, c = torch.unique_consecutive(word_id, return_counts=True)
        embed_list = []
        h_i = 0
        for i in range(c.shape[0]):
            if c[i] > 1:
                to_avg = hidden[h_i:h_i+c[i]]
                avgd = torch.mean(to_avg, dim=0) # d dimensional
                embed_list.append(avgd)
                h_i += c[i]
            else:
                embed_list.append(hidden[h_i])
                h_i += 1

        return torch.stack(embed_list)


    def re_pad(self, hidden_state, max_len):
        new_hidden_state = []
        for i in range(len(hidden_state)):
            s = hidden_state[i].shape[0] # length of the current sequence
            h_new = nn.functional.pad(hidden_state[i], (0, 0, 0, max_len-s)) # pad to max_len
            new_hidden_state.append(h_new)

        new_hidden_state = torch.stack(new_hidden_state)
        return new_hidden_state

    
    def forward(self, hidden_state, word_ids, label_mask, max_len):

        # hidden_state -> b, s', d
        # word_ids -> b, s' # -100 for both padding and special tokens
        # attention mask -> b, s' # will not mask out special tokens

        new_hidden_state = []

        for i in range(hidden_state.shape[0]):
            h = hidden_state[i] # s', d
            # del -100 tokens
            word_id = word_ids[i][word_ids[i] != -100]
            # ids for which we dont want the embedding (special tokens and pad)
            del_ids = (word_ids[i] == -100).nonzero(as_tuple=True)[0].tolist()
            # del those embeddings 
            h_new = self.del_embeds(h, del_ids) # s+duplicates, d

            # average over subword embeddings
            h_final = self.avg_embed(h_new, word_id) # s, d (s seq length for the example)

            new_hidden_state.append(h_final)

        # pad to max length in batch
        new_hidden_state = self.re_pad(new_hidden_state, max_len) # b, s, d 


        transformed = torch.matmul(new_hidden_state, self.proj) # b,s,r
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2) # b, s, 1, r
        transformed = transformed.expand(-1, -1, seqlen, -1) # b, s, s, r
        transposed = transformed.transpose(1,2) # b, s, s, r
        diffs = transformed - transposed # b, s, s, r
        squared_diffs = diffs.pow(2) # b, s, s, r
        squared_distances = torch.sum(squared_diffs, -1) # b, s, s

        # mask out useless values to zero (maybe later?)
        squared_distances = squared_distances * label_mask
        

        return squared_distances


# L1 loss for distance matrices
class L1DistanceLoss(nn.Module):

    def __init__(self, args):
        super(L1DistanceLoss, self).__init__()
        self.args = args
        self.loss = nn.L1Loss(reduction='none') 

    def forward(self, predictions, labels, label_mask, lens):
        # computes L1 loss on distance matrices.

        labels = labels * label_mask
        loss = self.loss(predictions, labels)
        summed_loss = torch.sum(loss, dim=(1,2)) # sum for each sequence
        loss = torch.sum(torch.div(summed_loss, lens.pow(2)))
        return loss


    

# depth probe class
class DepthProbe:
    pass



if __name__ == '__main__':

    argp = ArgumentParser()

    ## Training Args ##
    # model
    argp.add_argument('--model_name', type=str, default='xlm-roberta-base')
    # probe dimension
    argp.add_argument('--probe_rank', type=int, default=None)
    # max length
    argp.add_argument('--max_length', type=int, default=256)
    # training task
    argp.add_argument('--task', type=str, default='node_distance')
    # train batch size
    argp.add_argument('--train_batch_size', type=int, default=16)
    # eval batch size
    argp.add_argument('--eval_batch_size', type=int, default=8)
    # embedding layer to use for projection
    argp.add_argument('--embed_layer', type=int, default=6)


    ## Data Args ##
    # dataset
    argp.add_argument('--dataset_name', type=str, default='universal_dependencies')
    # language
    argp.add_argument('--lang', type=str, default='en')
    # treebank config
    argp.add_argument('--config', type=str, default='en_pud')
    # multiple configs
    argp.add_argument('--config_list', type=list)
    # process data anyway
    argp.add_argument('--process_data', default=False, action=argparse.BooleanOptionalAction)
    # save and overwrite processed data
    argp.add_argument('--save_processed_data', default=False, action=argparse.BooleanOptionalAction)
    # processed data directory
    argp.add_argument('--processed_data_dir', type=str, default=root+'/data/processed/UD/')


    ## Other Hyperparameters ##
    # seed
    argp.add_argument('--seed', type=int, default=42)



    # parse cli arguments
    args = argp.parse_args() 

    # seed
    set_seed(args.seed)

    # dataset and language
    print('dataset : {}'.format(args.dataset_name))
    print('language set to {}'.format(args.lang))

    # task
    if args.task is None:
        raise ValueError(
            f"specify training task with --task, from ({tasks})"
        )
    if args.task not in tasks:
        raise ValueError(
            f"task not supported yet. Choose from ({tasks})"
        )
    print('task set to {}'.format(args.task))

    # treebank config
    # add support for merging multiple treebanks here
    if args.config_list is not None:
        pass
    else:
        print('using config {}'.format(args.config))


    # tokenizer
    print('loading tokenizer {}'.format(args.model_name))
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)


    # UD processing class
    ud = UD(args, tokenizer)

    # check if proecssed data exists
    processed_data_dir = args.processed_data_dir+args.config+'_'+args.task
    if not args.process_data and os.path.isdir(processed_data_dir) and len(os.listdir(processed_data_dir)) > 0:
        print('loading processed data from {}'.format(processed_data_dir))
        dataset = load_from_disk(processed_data_dir)
    
    else:
        # get data and process 
        print('processing data')
        dataset = ud.process_data(args)

        # save to disk
        if args.save_processed_data:
            print('saving processed data')
            dataset.save_to_disk(processed_data_dir)
            print('saved')

    # dataset -> input_ids, attention_mask, labels, label_mask, word_ids


    print('data collator with padding')
    # data colator -> input_ids, attention_mask, labels, label_mask, word_ids
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # data loader
    dataloader = DataLoader(
        dataset["test"],
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )

    # model
    # a base model without any specific head
    model = XLMRobertaModel.from_pretrained(args.model_name)

    # probe
    print('intializing probe for task : {}'.format(args.task))
    # need to load model first for this
    probe = DistanceProbe(model.config.hidden_size, args.probe_rank)


    batch = next(iter(dataloader))

    inputs = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']  # -100 pad
    label_mask = batch['label_mask']
    lens = batch['lens']

    outputs = model(
        input_ids=inputs,
        attention_mask=attention_mask,
        output_hidden_states=True
    )


    rep = outputs.last_hidden_state
    pred_dist = probe(rep, batch['word_ids'], label_mask, label_mask.shape[-1])


    # compute loss 
    l1 = L1DistanceLoss(args)
    loss = l1(pred_dist, labels, label_mask, lens)
    print(loss)





    

