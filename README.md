# train probe
python train_probe.py --model_name facebook/xlm-v-base --embed_layer 6 --probe_rank 128 --overwrite_output_dir --do_train --do_eval --all_configs --process_data --save_processed_data --early_stop

python train_probe.py --model_name facebook/xlm-v-base --embed_layer 6 --num_train_epochs 200  --train_batch_size 2 --eval_batch_size 2 --overwrite_output_dir --do_train --do_eval --all_configs --process_data --save_processed_data --early_stop

# train probe for adapter
python train_probe_adapter.py --embed_layer 6 --lang is --train_batch_size 8 --eval_batch_size 4 --load_pretrained_model --pretrained_model models/finetuned/wikiann_8 --overwrite_output_dir --do_train --do_eval --all_configs --process_data --save_processed_data --early_stop

# eval probe with adapter
python train_probe_adapter.py --embed_layer 6  --load_pretrained_model --pretrained_model models/finetuned/wikiann_8 --load_pretrained_probe --pretrained_probe models/probes/node_distance_layer_6_98  --do_eval --all_configs --process_data --save_processed_data 

# eval probe with tgt adapter
python train_probe_adapter.py --embed_layer 6 --lang is --eval_batch_size 4 --load_pretrained_model --pretrained_model models/finetuned/wikiann_8 --load_pretrained_probe --pretrained_probe models/probes/node_distance_layer_6_98  --tgt_adapter --do_eval --all_configs --process_data --save_processed_data 

# train probe for adapter with tgt adapter
python train_probe_adapter.py --embed_layer 6 --lang is --train_batch_size 8 --eval_batch_size 4 --load_pretrained_model --pretrained_model models/finetuned/wikiann_8 --tgt_adapter --overwrite_output_dir --do_train --do_eval --all_configs --process_data --save_processed_data --early_stop

# train ner adapter
python ner_adapter.py --process_data --save_processed_data --overwrite_output_dir --do_train --early_stop