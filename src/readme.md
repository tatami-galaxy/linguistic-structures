python train_probe.py --num_train_epochs 20 --all_config --process_data --save_processed_data --overwrite_output_dir --do_train --do_eval

python probe.py --num_train_epochs 100 --overwrite_output_dir --do_train --all_configs --process_data --save_processed_data