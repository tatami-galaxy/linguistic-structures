# imports
import os
from os.path import dirname, abspath
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, XLMRobertaModel
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
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

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

    tokenizer: AutoTokenizer
    padding: bool = True
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"]} for feature in features]
        label_features = [{"true_dist": feature["true_dist"]} for feature in features]


        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            return_tensors=self.return_tensors,
        )

        # pad true_dist
        # label_features -> list (len = batch_size) of dicts
        # dict -> 'true_dist' : list (len = seq_len x seq_len) of distances (float)
        # tokenizer.pad_token_id = 1
        lens = [len(x['true_dist']) for x in label_features]
        max_len = max(lens)
        attentions = []
        # extending to the longest example with pad token ids
        for i in range(len(lens)): # len(lens) = batch_size 
            attentions.append([1]*lens[i]) # 1s for true length in mask
            # extend to max len with pad token id (1)
            label_features[i]['true_dist'].extend([tokenizer.pad_token_id]*(max_len-lens[i])) 
            # extend attention to max len with 0s
            attentions[i].extend([0]*(max_len-lens[i]))
        
        # convert to tensors
        batch["labels"] = torch.Tensor([label_features[i]['true_dist'] for i in range(len(lens))])
        batch["label_attention_mask"] = torch.Tensor([attentions[i] for i in range(len(lens))])

        return batch


# UD class

class UD:

    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
            # convert to numpy and flatten
            dist_matrix_f = np.array(dist_matrix).flatten()

            dists.append(dist_matrix_f.tolist())

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
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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

    # get
    def forward(self, input_ids):
        transformed = torch.matmul(input_ids, self.proj) # b,s,r
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2) # b, s, 1, r
        transformed = transformed.expand(-1, -1, seqlen, -1) # b, s, s, r
        transposed = transformed.transpose(1,2) # b, s, s, r
        diffs = transformed - transposed # b, s, s, r
        squared_diffs = diffs.pow(2) # b, s, s, r
        squared_distances = torch.sum(squared_diffs, -1) # b, s, s
        return squared_distances


# L1 loss for distance matrices
class L1DistanceLoss(nn.Module):

    def __init__(self, args):
        super(L1DistanceLoss, self).__init__()
        self.args = args
        self.word_pair_dims = (1,2)

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.

        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.

        Args:
        predictions: A pytorch batch of predicted distances
        label_batch: A pytorch batch of true distances
        length_batch: A pytorch batch of sentence lengths

        Returns:
        A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.args['device'])
        return batch_loss, total_sents

    

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


    # UD processing class
    ud = UD(args)

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


    # tokenizer
    print('loading tokenizer {}'.format(args.model_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print('data collator with padding')
    # data colator
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
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        output_hidden_states=True)

    # which layer to use?
    # compute loss
    # need to un-flatten true distance matrix

    rep = outputs.last_hidden_state
    print(rep.shape)
    pred_dist = probe(rep)
    print(pred_dist.shape)


    

