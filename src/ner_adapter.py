from transformers import AutoTokenizer, AutoConfig, XLMRobertaAdapterModel, set_seed
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
from transformers import AdapterConfig
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from datasets import load_metric
import numpy as np
from transformers.adapters.composition import Stack
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import os
import sys
from os.path import dirname, abspath
import argparse
from argparse import ArgumentParser

# the labels for the NER task and the dictionaries to map the to ids or 
# the other way around
labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
id_2_label = {id_: label for id_, label in enumerate(labels)}
label_2_id = {label: id_ for id_, label in enumerate(labels)}

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'linguistic-structures':
    root = dirname(root)


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
    

# tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    text_column_name = "tokens"
    label_column_name = "ner_tags"
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=False,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.  
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# postprocessing function for eval
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    label_list = id_2_label

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_labels, true_predictions

    

if __name__ == '__main__':

    argp = ArgumentParser()

    # model
    argp.add_argument('--model_name', type=str, default='xlm-roberta-base')
    # epochs
    argp.add_argument('--num_train_epochs', type=int, default=30)
    # learning rate
    argp.add_argument('--learning_rate', type=float, default=1e-4)  # 5e-5
    # warmup steps
    argp.add_argument('--num_warmup_steps', type=int, default=0) 
    # train batch size
    argp.add_argument('--train_batch_size', type=int, default=16)
    # eval batch size
    argp.add_argument('--eval_batch_size', type=int, default=8)
    # probe save directory
    #argp.add_argument('--output_dir', type=str, default=root+'/models/probes/')
    # overwrite output dir
    #argp.add_argument('--overwrite_output_dir', default=False, action=argparse.BooleanOptionalAction)
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
    # early stop
    argp.add_argument('--early_stop', default=False, action=argparse.BooleanOptionalAction)
    # early stop patience
    argp.add_argument('--patience', type=int, default=7)
    # early stop min delta
    argp.add_argument('--min_delta', type=float, default=0.5)
    # dataset
    argp.add_argument('--dataset_name', type=str, default='wikiann')
    # language
    argp.add_argument('--src_lang', type=str, default='en')
    argp.add_argument('--tgt_lang', type=str, default=None)
    # process data anyway
    argp.add_argument('--process_data', default=False, action=argparse.BooleanOptionalAction)
    # save and overwrite processed data
    argp.add_argument('--save_processed_data', default=False, action=argparse.BooleanOptionalAction)
    # processed data directory
    argp.add_argument('--processed_data_dir', type=str, default=root+'/data/processed/wikiann/')
    # seed
    argp.add_argument('--seed', type=int, default=42)



    # parse and check cli arguments #
    args = argp.parse_args() 

    # seed
    set_seed(args.seed)

    # model and tokenizer

    print('model : {}'.format(args.model_name))

    config = AutoConfig.from_pretrained(args.model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = XLMRobertaAdapterModel.from_pretrained(args.model_name, config=config)

    # load source language adapter
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    model.load_adapter(args.src_lang+"/wiki@ukp", config=lang_adapter_config) # leave_out=[11])
    # add a new task adapter for NER
    model.add_adapter("ner")  # name = ner
    # NER tagging head
    model.add_tagging_head("ner_head", num_labels=len(labels))  # name = ner_head
    # train only task adapter
    model.train_adapter(["ner"])
    # stack on top of src lang adapter
    model.active_adapters = Stack(args.src_lang, "ner")

    # dataset
    # processed_data_dir different from args.processed_data_dir
    processed_data_dir = args.processed_data_dir+args.src_lang+'_'+args.dataset_name
    if not args.process_data and os.path.isdir(processed_data_dir) and len(os.listdir(processed_data_dir)) > 0:
        print('loading processed data from {}'.format(processed_data_dir))
        dataset = load_from_disk(processed_data_dir)
    else:
        # get data and process 
        print('downloading and processing data')
        dataset = load_dataset(args.dataset_name, args.src_lang)
        # process dataset
        dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        # save to disk
        if args.save_processed_data:
            print('saving processed data')
            dataset.save_to_disk(processed_data_dir)
    
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))

    if args.max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(args.max_test_samples))


    # data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # data loaders
    train_dataloader = DataLoader(
        dataset['train'],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=args.train_batch_size,
    )
    eval_dataloader = DataLoader(
        dataset['validation'],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=args.eval_batch_size,
    )

    # metric
    metric = load_metric("seqeval")

    # optimizer
    # training probe only (not model)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # train steps
    num_training_steps = args.num_train_epochs * len(train_dataloader)
    # eval steps
    num_eval_steps = len(eval_dataloader)
    # scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # setup early stopping
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    if args.do_train: # whether to train or not
        print('train steps : {}'.format(num_training_steps))
        progress_bar = tqdm(range(num_training_steps))
        model.train()
        print('training')
        if not args.overwrite_output_dir:
            print("--overwrite_output_dir set to False. Won't save trained probe!")

        for epoch in range(args.num_train_epochs):
            for batch in train_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            print('train loss : {}'.format(train_loss/len(train_dataloader)))
            print('calculating eval loss')


