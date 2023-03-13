# train probe
python probe.py --model_name facebook/xlm-v-base --embed_layer 6 --probe_rank 128 --overwrite_output_dir --do_train --do_eval --all_configs --process_data --save_processed_data --early_stop

# train ner adapter
python ner_adapter.py --process_data --save_processed_data --overwrite_output_dir --do_train --early_stop