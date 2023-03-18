# imports
import os
import sys
from os.path import dirname, abspath

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'linguistic-structures':
    root = dirname(root)

# append to path for loading class
sys.path.append(root+'/src/')

from typing import Any, Dict, List, Optional
from transformers import XLMRobertaTokenizerFast, XLMRobertaAdapterModel, AdapterConfig, AutoConfig
from transformers.adapters.composition import Stack
from transformers import AdamW
from transformers import get_scheduler
from transformers import set_seed
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
import argparse
from argparse import ArgumentParser
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from scipy.sparse.csgraph import floyd_warshall
from dataclasses import dataclass
from tqdm.auto import tqdm
from utils import DistanceProbe, L1DistanceLoss, Metrics

# the labels for the NER task and the dictionaries to map the to ids or 
# the other way around
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}



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

        tokens = [{"tokens": feature["tokens"]} for feature in features]

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
        batch["sentences"] = tokens

        return batch


# UD class

class UD:

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    # dataset mapping functions
    # get pairwise distances for sentences
    # dist_matrix specifies distance between nodes, not edges
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


    def merge_treebanks(self, dataset_name, config_list):

        dataset = DatasetDict()
        config = config_list.pop(0) # first config, assume has train, val, test

        dataset['train'] = load_dataset(dataset_name, config, split='train')
        dataset['validation'] = load_dataset(dataset_name, config, split='validation')
        dataset['test'] = load_dataset(dataset_name, config, split='test')

        for config in config_list:
            data = load_dataset(dataset_name, config)
            keys = data.keys()
            for key in keys:
                dataset[key] = concatenate_datasets([dataset[key], data[key]])

        return dataset


    # process dataset given training task
    def process_data(self, args):

        if args.all_configs and args.config_list is not None:
            print('using config {}'.format(args.config_list))
            ## merge UD treebanks ##
            dataset = self.merge_treebanks(args.dataset_name, args.config_list)
            # shuffling
            dataset['train'] = dataset['train'].shuffle(seed=args.seed)
            dataset['validation'] = dataset['validation'].shuffle(seed=args.seed)
            dataset['test'] = dataset['test'].shuffle(seed=args.seed)

        else:
            print('using config {}'.format(args.config))  # assume to have a default value
            dataset = load_dataset(args.dataset_name, args.config)

        # filter examples with None in head
        dataset = dataset.filter(lambda example: all(h.isdigit() for h in example['head']))

        # filter very small sentences #
        dataset = dataset.filter(lambda example: len(example['head']) >= args.min_length)

        # number of tokens in UD <= number of tokens in tokenized text
        # need to map tokenized tokens to UD tokens
        if args.task == 'node_distance':
            # inputs_ids, attention_mask, true_dists
            dataset = dataset.map(self.distance_map, batched=True) #remove_columns=dataset['test'].column_names)
            return dataset



# pytorch early stopping
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



if __name__ == '__main__':

    argp = ArgumentParser()

    ## Training Args ##
    # model
    argp.add_argument('--model_name', type=str, default='xlm-roberta-base')
    # probe dimension
    argp.add_argument('--probe_rank', type=int, default=None) # 128 or lower than dim
    # max length
    argp.add_argument('--max_length', type=int, default=256)
    # training task
    argp.add_argument('--task', type=str, default='node_distance')
    # epochs
    # how long to train?
    # what's the stopping crieteria?
    argp.add_argument('--num_train_epochs', type=int, default=100)
    # learning rate
    argp.add_argument('--learning_rate', type=float, default=3e-4)  # 5e-5
    # train batch size
    argp.add_argument('--train_batch_size', type=int, default=16)
    # eval batch size
    argp.add_argument('--eval_batch_size', type=int, default=8)
    # embedding layer to use for projection
    argp.add_argument('--embed_layer', type=int, default=6)
    # probe save directory
    argp.add_argument('--output_dir', type=str, default=root+'/models/probes/')
    # overwrite output dir
    argp.add_argument('--overwrite_output_dir', default=False, action=argparse.BooleanOptionalAction)
    # train
    argp.add_argument('--do_train', default=False, action=argparse.BooleanOptionalAction)
    # eval
    argp.add_argument('--do_eval', default=False, action=argparse.BooleanOptionalAction)
    # max train size
    argp.add_argument('--max_train_samples', type=int, default=None)
    # max eval size
    argp.add_argument('--max_eval_samples', type=int, default=None)
    # max test size
    argp.add_argument('--max_test_samples', type=int, default=None)
    # pretrained probe
    argp.add_argument('--load_pretrained_probe', default=False, action=argparse.BooleanOptionalAction)
    # pretrained probe location
    argp.add_argument('--pretrained_probe', type=str, default=None)
    # pretrained model
    argp.add_argument('--load_pretrained_model', default=False, action=argparse.BooleanOptionalAction)
    # pretrained model location wrt project root
    argp.add_argument('--pretrained_model', type=str, default=None)
    # switch lang aapter
    argp.add_argument('--tgt_adapter', default=False, action=argparse.BooleanOptionalAction)
    # early stop
    argp.add_argument('--early_stop', default=False, action=argparse.BooleanOptionalAction)
    # early stop patience
    argp.add_argument('--patience', type=int, default=7)
    # early stop min delta
    argp.add_argument('--min_delta', type=float, default=0.5)


    ## Data Args ##
    # dataset
    argp.add_argument('--dataset_name', type=str, default='universal_dependencies')
    # dataset language
    argp.add_argument('--lang', type=str, default='en')
    # adapter src language
    argp.add_argument('--src_lang', type=str, default='en')
    # adapter src language
    argp.add_argument('--tgt_lang', type=str, default='is')
    # treebank config
    # in order to use only this need to check what configs are in processed data dir
    # if its not this, pass in --process_data
    argp.add_argument('--config', type=str, default='en_pud') # make sure this is not None
    # multiple configs
    argp.add_argument(
        '--config_list',
        type=list[str],
        #default=['is_icepahc', 'is_pud'])
        default=['en_ewt', 'en_gum', 'en_lines', 'en_partut', 'en_pronouns', 'en_pud'])
    # use all configs
    argp.add_argument('--all_configs', default=False, action=argparse.BooleanOptionalAction)
    # minimum sentence length
    argp.add_argument('--min_length', type=int, default=4)
    # maximum sentence length
    # do we need this?
    #argp.add_argument('--max_length', type=int, default=50)
    # process data anyway
    argp.add_argument('--process_data', default=False, action=argparse.BooleanOptionalAction)
    # save and overwrite processed data
    argp.add_argument('--save_processed_data', default=False, action=argparse.BooleanOptionalAction)
    # processed data directory
    argp.add_argument('--processed_data_dir', type=str, default=root+'/data/processed/UD/')


    ## Other Hyperparameters ##
    # seed
    argp.add_argument('--seed', type=int, default=42)



    # parse and check cli arguments #
    args = argp.parse_args() 

    # seed
    set_seed(args.seed)
    # dataset and language
    print('dataset : {}'.format(args.dataset_name))
    print('language set to {}'.format(args.lang))
    # layer
    print('layer set to {}'.format(args.embed_layer))
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
    # output directory
    if args.output_dir is None:
        raise ValueError(
            f"pass in output directory"
        )

    # tokenizer #
    print('loading tokenizer {}'.format(args.model_name))
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    # data processing
    # UD processing class
    ud = UD(args, tokenizer)
    # check if proecssed data exists
    processed_data_dir = args.processed_data_dir+args.lang+'_'+args.task
    if not args.process_data and os.path.isdir(processed_data_dir) and len(os.listdir(processed_data_dir)) > 0:
        print('loading processed data from {}'.format(processed_data_dir))
        dataset = load_from_disk(processed_data_dir)
        with open(args.processed_data_dir+'data_config.txt') as f:
            data_config = f.readline()
            print('data config : {}'.format(data_config)) 
    else:
        # get data and process 
        print('processing data')
        dataset = ud.process_data(args)
        # save to disk
        if args.save_processed_data:
            print('saving processed data')
            dataset.save_to_disk(processed_data_dir)
            with open(args.processed_data_dir+'data_config.txt', 'w') as f:
                if args.all_configs and args.config_list is not None:
                    f.write(', '.join(args.config_list))
                else:
                    f.write(args.config) # assume to have default value
            print('saved')
    
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))

    if args.max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(args.max_test_samples))

    # dataset -> input_ids, attention_mask, word_ids, true_dist


    # data collator, data loader
    print('data collator with padding')
    # data colator -> input_ids, attention_mask, labels, label_mask, word_ids
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data loaders
    train_dataloader = DataLoader(
        dataset['train'],
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )
    eval_dataloader = DataLoader(
        dataset['validation'],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )


    # model
    print('model : {}'.format(args.model_name))

    #config = AutoConfig.from_pretrained(args.model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    # load source language adapter
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)

    # load finetuned model
    if args.load_pretrained_model:
        if args.pretrained_model is None:
            raise ValueError(
            f"pass in pretrained model (--pretrained_model)"
        )
        # absolute path of pretrained model
        pretrained_model = root+'/'+args.pretrained_model
        print('loading pretrained model')
        # load pretrained model
        model = XLMRobertaAdapterModel.from_pretrained(pretrained_model)

    else: 
        print('initializing adapter model')
        config = AutoConfig.from_pretrained(
            args.model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
        model = XLMRobertaAdapterModel.from_pretrained(args.model_name, config=config)

        # load source language adapter
        #lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        model.load_adapter(args.src_lang+"/wiki@ukp", config=lang_adapter_config) # leave_out=[11])
        # add a new task adapter for NER
        model.add_adapter("ner")  # name = ner
        # NER tagging head
        model.add_tagging_head("ner_head", num_labels=len(labels))  # name = ner_head


    if args.tgt_adapter:
        model.load_adapter(
            args.tgt_lang+"/wiki@ukp",
            config=lang_adapter_config,
            model_name=args.model_name) # leave_out=[11])
        # stack on top of tgt lang adapter
        model.active_adapters = Stack(args.tgt_lang, "ner")
    else:
        # stack on top of src lang adapter
        model.active_adapters = Stack(args.src_lang, "ner")


    # probe
    print('intializing probe for task : {}'.format(args.task))
    # need to load model first for this
    probe = DistanceProbe(model.config.hidden_size, args.probe_rank)
    if args.load_pretrained_probe:
        if args.pretrained_probe is None:
            raise ValueError(
            f"pass in pretrained probe (--pretrained_probe)"
        )
        print('loading pretrained probe')
        if torch.cuda.is_available():
            probe.load_state_dict(torch.load(args.pretrained_probe)) 
        else:
            probe.load_state_dict(torch.load(args.pretrained_probe, map_location=torch.device('cpu')))


    # loss function
    l1 = L1DistanceLoss(args)

    # metric
    metric = Metrics(args)

    # optimizer
    # training probe only (not model)
    optimizer = AdamW(probe.parameters(), lr=args.learning_rate)
    # train steps
    num_training_steps = args.num_train_epochs * len(train_dataloader)
    # eval steps
    num_eval_steps = len(eval_dataloader)
    # scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    probe.to(device)


    # train loop
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    if args.do_train: # whether to train or not
        print('train steps : {}'.format(num_training_steps))
        progress_bar = tqdm(range(num_training_steps))
        print('training')
        if not args.overwrite_output_dir:
            print("--overwrite_output_dir set to False. Won't save trained probe!")

        for epoch in range(args.num_train_epochs):
            probe.train()
            ## training ##
            train_loss = 0
            for batch in train_dataloader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)  # -100 pad
                label_mask = batch['label_mask'].to(device)
                word_ids = batch['word_ids'].to(device)
                lens = batch['lens'].to(device)

                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                #rep = outputs.last_hidden_state 
                # rep is after passing through task adapter of the layer
                rep = outputs.hidden_states[args.embed_layer + 1] # 0 : embedding layer

                pred_dist = probe(rep, word_ids, label_mask, label_mask.shape[-1])

                # loss
                loss = l1(pred_dist, labels, label_mask, lens)
                train_loss += loss.item()

                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            print('train loss : {}'.format(train_loss/len(train_dataloader)))
            print('calculating eval loss')

            ## evaluation ##
            val_loss = 0
            probe.eval()
            #eval_bar = tqdm(range(num_eval_steps))
            for batch in eval_dataloader:
                with torch.no_grad():
                    inputs = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)  # -100 pad
                    label_mask = batch['label_mask'].to(device)
                    word_ids = batch['word_ids'].to(device)
                    lens = batch['lens'].to(device)

                    outputs = model(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )

                    #rep = outputs.last_hidden_state 
                    # rep is after passing through task adapter of the layer
                    rep = outputs.hidden_states[args.embed_layer + 1] # 0 : embedding layer
                    
                    pred_dist = probe(rep, word_ids, label_mask, label_mask.shape[-1])

                    # loss
                    val_loss += l1(pred_dist, labels, label_mask, lens).item()

            print('val loss at epoch {} : {}'.format(epoch, val_loss/len(eval_dataloader)))

            if early_stopper.early_stop(val_loss):
                print('early stop at epoch {}'.format(epoch))
                if args.overwrite_output_dir:
                    print('saving final probe')
                    checkpoint_str = args.output_dir+'/'+args.task+'_layer_'+str(args.embed_layer)+'_'+str(epoch)
                    torch.save(probe.state_dict(), checkpoint_str)
                break

            if args.overwrite_output_dir:
                print('saving probe')
                checkpoint_str = args.output_dir+'/'+args.task+'_layer_'+str(args.embed_layer)+'_'+str(epoch)
                torch.save(probe.state_dict(), checkpoint_str)

    else:
        print('did not train. set --do_train to train')

    if args.do_eval:
        print('eval steps : {}'.format(num_eval_steps))
        print("evaluating")
        probe.eval()
        eval_bar = tqdm(range(num_eval_steps))
        #print("generating distance image")
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)  # -100 pad
                label_mask = batch['label_mask'].to(device)
                word_ids = batch['word_ids'].to(device)
                lens = batch['lens'].to(device)

                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                #rep = outputs.last_hidden_state ## change to layer rep
                # rep is after passing through task adapter of the layer
                rep = outputs.hidden_states[args.embed_layer + 1] # 0 : embedding layer
                pred_dist = probe(rep, word_ids, label_mask, label_mask.shape[-1])

                # spearman for each batch
                sentences = batch["sentences"]
                metric.add_spearman(pred_dist, labels, label_mask, sentences)

                # uuas
                metric.add_uuas(pred_dist, labels, label_mask, sentences)


                eval_bar.update(1)

        metric.compute_spearman()
        metric.compute_uuas()
        
        for key, val in metric.results.items():
            print('{} : {}'.format(key, val))

    else:
        print('did not eval. set --do_eval to train')


    print('done.')







    

