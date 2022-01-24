# Folder:


# Quick start
1. Training: 
Mnist:

`python main.py --use_se --num_initial_channel 32 --num_process_blocks 1 --num_preprocess_cells 2 --num_postprocess_cells 2 --num_cell_per_group_enc 2 --num_cell_per_group_dec 2 --epochs 3 --model_path ./model_output/cesm_test --batch_size 2 --num_ipus 2`

CESM: (data can be downloaded at SDRBench)

`python main.py --use_se --num_initial_channel 32 --num_process_blocks 3 --num_preprocess_cells 2 --num_postprocess_cells 2 --num_cell_per_group_enc 2 --num_cell_per_group_dec 2 --epochs 3 --model_path ./model_output/cesm_test --data_path ../data --batch_size 2 --num_ipus 2`


2. Evaluating: ``

3. Generating images: ``


# GitHub URL
**[]()**

# License
This program is created by [Hieu Le](https://github.com/hieutrungle)
