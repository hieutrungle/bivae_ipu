import os
import numpy as np
import shutil
from typing import *

"""Data Preprocessing Helper Functions"""

def get_data_info(data:np.ndarray):
    print(f"data shape: {data.shape}")
    print(f"maximum value: {data.max()}")
    print(f"minimum value: {data.min()}\n")


# Get all file names in directory
def get_filename(parent_dir:str, file_extension:str)->str:
    filenames = []
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (file_extension in filename):
                filenames.append(os.path.join(parent_dir, filename))
    return filenames

# Create new directory if not exist
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create folders to store training information
def mkdir_storage(model_dir, resume={}):
    if os.path.exists(os.path.join(model_dir, 'summaries')):
        if len(resume) == 0:
        # val = input("The model directory %s exists. Overwrite? (y/n) " % model_dir)
        # print()
        # if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))
    
    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    mkdir_if_not_exist(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    mkdir_if_not_exist(checkpoints_dir)
    return summaries_dir, checkpoints_dir


def get_folder_size(start_path:str='.')->int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g

def get_model_arch(arch_type):
    if arch_type == 'res_bnswish':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['ar_nn'] = ['res_bnswish']
    elif arch_type == 'res_mbconv':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['mconv_e6k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e6k5g0']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e6k5g0']
    elif arch_type == 'res_wnelu':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_enc'] = ['res_wnelu', 'res_elu']
        model_arch['normal_dec'] = ['mconv_e3k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e3k5g0']
        model_arch['normal_pre'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_pre'] = ['res_wnelu', 'res_elu']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e3k5g0']
    return model_arch