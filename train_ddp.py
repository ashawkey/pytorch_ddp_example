import os
import json
import random
import collections

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import time

import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp

from sklearn.model_selection import StratifiedKFold

from utils import *
from models import Model
from provider import DataRetriever

# hyperparams

SEED = 42
EPOCHS = 5

seed_everything(SEED)


def main(rank, world_size, fold):
    
    # initialize
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank) # important! solved imbalanced memory at GPU0, ref: https://discuss.pytorch.org/t/how-to-balance-gpu-memories-in-ddp/93170
    print(f'[DDP] initialized for rank {rank} / {world_size}, fold {fold}')
    
    # load data
    train_df = pd.read_csv("../input/g2net-gravitational-wave-detection/training_labels.csv")
    #train_df = train_df[:1000]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(train_df, train_df['target']))
    df_train = train_df.iloc[splits[fold][0]]
    df_valid = train_df.iloc[splits[fold][1]]
    print(f'[DATA] FOLD = {fold}, ', df_train.shape, df_valid.shape)

    # create dataset
    train_data_retriever = DataRetriever(df_train["id"].values, df_train["target"].values)
    valid_data_retriever = DataRetriever(df_valid["id"].values, df_valid["target"].values)

    # create dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_retriever, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED, drop_last=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data_retriever, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    # the batch_size and num_workers are per-GPU !
    train_loader = torch.utils.data.DataLoader(train_data_retriever, batch_size=16, pin_memory=True, num_workers=4, sampler=train_sampler)  
    valid_loader = torch.utils.data.DataLoader(valid_data_retriever, batch_size=16, pin_memory=True, num_workers=4, sampler=valid_sampler)

    # create trainer
    model = Model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = lambda m: torch.optim.Adam(m.parameters(), lr=0.001)
    scheduler = lambda o: optim.lr_scheduler.StepLR(o, step_size=3, gamma=0.1)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=len(train_loader) * EPOCHS)

    trainer = Trainer(f'fold{fold}', model, local_rank=rank, world_size=world_size, fp16=True, 
                      optimizer=optimizer, criterion=criterion, lr_scheduler=scheduler, metrics=[ROCMeter(),], use_checkpoint='latest', eval_interval=1,
                      scheduler_update_every_step=False)
    
    # train
    trainer.train(train_loader, valid_loader, EPOCHS)

    # clean
    dist.destroy_process_group()


if __name__ == '__main__':

    WORLD_SIZE = 4 # 4
    
    for fold in range(5):
        print(f'====================')
        mp.spawn(main, args=(WORLD_SIZE, fold), nprocs=WORLD_SIZE, join=True)
        print(f'====================')
