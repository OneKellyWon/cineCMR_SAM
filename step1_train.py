#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation
# 
# You should prepare the following before running this step. Please refer to the `example_data/data` folder for guidance:
# 
# 1. **SAX data** including image and manual segmentation (for training)
#    - you want to prepare the SAX data as a 4D array [x,y,time_frame,slice_num] saved as a nii file. in our study we sample 15 time frames as default
#    - please refer ```example_data/data/ID_0002``` as SAX reference  
# 
# 2. **LAX data** including image and manual segmentation (for training), if you need to train for LAX segmentation as well
#    - you want to prepare the LAX data as a 3D array [x,y,time_frame]. As aforementioned, time frame is default to 15
#    - please refer ```example_data/data/ID_0085``` as LAX reference 
# 
# 3. **A patient list** that enumerates all your cases
#    - To understand the standard format, please refer to the file:  
#      `example_data/Patient_list/patient_list.xlsx`
#    - make sure column ***total_slice_num*** is correct for each case
# 
# 4. **Text prompts** that specifies the view type
#    - our model takes text prompt "SAX" or "LAX" to specify the view type 
#    - we use "CLIP" model to embed text prompts (code: ```dataset/CMR/clip_extractor.ipynb```)
#    - we have prepared the embedded feature in `example_data/data/text_prompt_clip`, please download to your local
# 
# 5. **Box prompts** that indicates the location of myocardium
#    - during the training, the box prompts are automatically generated based on manual segmentation so don't need to worry about it
# 
# 6. **Original SAM model**
#    - download from [this link](https://github.com/SekeunKim/MediViSTA?tab=readme-ov-file)
# 
# 
# ---
# 
# ### Docker environment
# Please use `docker`, it will build a pytorch-based container
# 

# In[ ]:


import os
import sys
sys.path.append('/gpfs/work/aac/yifansun2302')  ### remove this if not needed!
import numpy as np
import pandas as pd 
from tqdm import tqdm 
import random
from pathlib import Path
import nibabel as nb
import time

import argparse
from einops import rearrange
from natsort import natsorted
from madgrad import MADGRAD

import torch
import torch.backends.cudnn as cudnn
 
from cineCMR_SAM.utils.model_util import *
from cineCMR_SAM.segment_anything.model import build_model 
from cineCMR_SAM.utils.save_utils import *
from cineCMR_SAM.utils.config_util import Config
from cineCMR_SAM.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from cineCMR_SAM.train_engine import train_loop

import cineCMR_SAM.dataset.build_CMR_datasets as build_CMR_datasets
import cineCMR_SAM.functions_collection as ff
import cineCMR_SAM.get_args_parser as get_args_parser

main_path = '/gpfs/work/aac/yifansun2302/cineCMR_SAM'  # replace with your own path


# ### define parameters for this experiment
# The full setting can be find in ```get_args_parser.py```

# In[ ]:


# set experiment-specific parameters
trial_name = 'cineCMR_sam_trial' 

output_dir = os.path.join(main_path, 'example_data/models', trial_name)
ff.make_folder([os.path.join(main_path, 'example_data/models'), output_dir])

text_prompt = False # whether we need to input text prompt to specify the view types (LAX or SAX). True or False. default = True
box_prompt = False # whether we have the bounding box for myocardium defined by the user. False means no box, 'one' means one box at ED and 'two' means two boxes at ED and ES

pretrained_model = None # define your pre-trained model if any
start_epoch = 1
total_training_epochs = 1 # define total number of epochs


# In[4]:


# default
# preload the text prompt feature 
sax_text_prompt_feature = np.load(os.path.join(main_path,'example_data/data/text_prompt_clip/sax.npy'))
lax_text_prompt_feature = np.load(os.path.join(main_path,'example_data/data/text_prompt_clip/lax.npy'))

# define the original SAM model
original_sam = os.path.join( main_path, 'example_data/pretrained_sam/sam_vit_h_4b8939.pth')  # can also use vit_b or vit_l, but need to change the arguments in get_args_parser accordingly

args = get_args_parser.get_args_parser(text_prompt = text_prompt, 
                                       box_prompt = box_prompt, 
                                       pretrained_model = pretrained_model, 
                                       original_sam = original_sam, 
                                       start_epoch = start_epoch, 
                                       total_training_epochs = total_training_epochs,
                                       vit_type = "vit_h")
args = args.parse_args([])

# some other settings
cfg = Config(args.config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# ### define the training dataset (from SAX or/and LAX)

# In[5]:


# define SAX training data
patient_list_file_sax = os.path.join(main_path,'example_data/data/Patient_list/patient_list_sax.xlsx')
patient_index_list = np.arange(0,1,1)
dataset_train_sax = build_CMR_datasets.build_dataset(
        args,
        view_type = 'sax',
        patient_list_file = patient_list_file_sax, 
        index_list = patient_index_list, 
        text_prompt_feature = sax_text_prompt_feature,
        only_myo = True, 
        shuffle = True, 
        augment = True)

# define LAX training data
patient_list_file_lax = os.path.join(main_path,'example_data/data/Patient_list/patient_list_lax.xlsx')
patient_index_list = np.arange(0,1,1)
dataset_train_lax = build_CMR_datasets.build_dataset(
        args,
        view_type = 'lax',
        patient_list_file = patient_list_file_lax, 
        index_list = patient_index_list, 
        text_prompt_feature = lax_text_prompt_feature,
        only_myo = True, 
        shuffle = True, 
        augment = True)

dataset_train = [dataset_train_sax, dataset_train_lax]

'''Set up data loader for training and validation set'''
data_loader_train = []
for i in range(len(dataset_train)):
    data_loader_train.append(torch.utils.data.DataLoader(dataset_train[i], batch_size = 1, shuffle = False, pin_memory = True, num_workers = 0))


# ### load pre-trained SAM model (freeze SAM modules)

# In[6]:


# set model
model = build_model(args, device)

# set freezed and trainable keys
train_keys = []
freezed_keys = []
        
# load pretrained sam model vit_h
if args.model_type.startswith("sam"):
    if args.resume.endswith(".pth"):
        with open(args.resume, "rb") as f:
            state_dict = torch.load(f)
        try:
            model.load_state_dict(state_dict)
        except:
            if args.vit_type == "vit_h" or args.vit_type == "vit_l" or args.vit_type == "vit_b":
                new_state_dict = load_from(model, state_dict, args.img_size,  16, [7, 15, 23, 31])
               
            model.load_state_dict(new_state_dict)
        
        # freeze original SAM layers
        freeze_list = [ "norm1", "attn" , "mlp", "norm2"]  
                
        for n, value in model.named_parameters():
            if any(substring in n for substring in freeze_list):
                freezed_keys.append(n)
                value.requires_grad = False
            else:
                train_keys.append(n)
                value.requires_grad = True

## Select optimization method
optimizer = MADGRAD(model.parameters(), lr=args.lr) # momentum=,weight_decay=,eps=)
        
# Continue training model
if args.pretrained_model is not None:
    if os.path.exists(args.pretrained_model):
        print('loading pretrained model : ', args.pretrained_model)
        args.resume = args.pretrained_model
        finetune_checkpoint = torch.load(args.pretrained_model)
        model.load_state_dict(finetune_checkpoint["model"])
        optimizer.load_state_dict(finetune_checkpoint["optimizer"])
        torch.cuda.empty_cache()
else:
    print('new training\n')


# ### Training

# In[7]:


training_log = []
model_save_folder = os.path.join(output_dir, 'models'); ff.make_folder([output_dir, model_save_folder])
log_save_folder = os.path.join(output_dir, 'logs'); ff.make_folder([log_save_folder])

for epoch in range(args.start_epoch, args.start_epoch + args.total_training_epochs):
        print('training epoch:', epoch)

        if epoch % args.lr_update_every_N_epoch == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * args.lr_decay_gamma
        print('learning rate now:', optimizer.param_groups[0]["lr"])
        
        loss_scaler = NativeScaler()
            
        train_results = train_loop(
                model = model,
                data_loader_train  = data_loader_train,
                optimizer = optimizer,
                epoch = epoch, 
                loss_scaler = loss_scaler,
                args = args,
                inputtype = cfg.data.input_type)   
        
        loss, lossCE, lossDICE, sax_loss, sax_lossCE, sax_lossDICE, lax_loss, lax_lossCE, lax_lossDICE = train_results       
            
        print('in epoch: ', epoch, ' training average_loss: ', loss, ' average_lossCE: ', lossCE, ' average_lossDICE: ', lossDICE, ' sax_loss: ', sax_loss, ' sax_lossCE: ', sax_lossCE, ' sax_lossDICE: ', sax_lossDICE, ' lax_loss: ', lax_loss, ' lax_lossCE: ', lax_lossCE, ' lax_lossDICE: ', lax_lossDICE)
    
        # on_epoch_end:
        for k in range(len(dataset_train)):
            dataset_train[k].on_epoch_end()
    
        if  epoch % args.save_model_file_every_N_epoch == 0 or (epoch + 1) == args.start_epoch + args.total_training_epochs:
            checkpoint_path = os.path.join(model_save_folder,  'model-%s.pth' % epoch)
            to_save = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,}
            torch.save(to_save, checkpoint_path)

        training_log.append([epoch, optimizer.param_groups[0]["lr"], train_results[0], train_results[1], train_results[2], train_results[3], train_results[4], train_results[5], train_results[6], train_results[7], train_results[8]])
        df = pd.DataFrame(training_log, columns=['epoch', 'lr','average_loss', 'average_lossCE', 'average_lossDICE', 'sax_loss', 'sax_lossCE', 'sax_lossDICE', 'lax_loss', 'lax_lossCE', 'lax_lossDICE'])
        df.to_excel(os.path.join(log_save_folder, 'training_log.xlsx'), index=False)


# In[ ]:




