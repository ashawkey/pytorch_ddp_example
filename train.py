import os
import json
import random
import collections

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch

import time

from torch import nn

from sklearn.model_selection import StratifiedKFold

from utils import *
from models import Model
from provider import DataRetriever

# hyperparams

SEED = 42

seed_everything(SEED)

train_df = pd.read_csv("../input/g2net-gravitational-wave-detection/training_labels.csv")

# split 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(skf.split(train_df, train_df['target']))

for FOLD in range(5):
    df_train = train_df.iloc[splits[FOLD][0]]
    df_valid = train_df.iloc[splits[FOLD][1]]
    print(f'[INFO] FOLD = {FOLD}, ', df_train.shape, df_valid.shape)

    # dataset
    train_data_retriever = DataRetriever(df_train["id"].values, df_train["target"].values)
    valid_data_retriever = DataRetriever(df_valid["id"].values, df_valid["target"].values)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_data_retriever, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_data_retriever, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

    # model
    model = Model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trainer = Trainer(f'fold{FOLD}', model, optimizer=optimizer, criterion=criterion, fp16=True, lr_scheduler=scheduler, metrics=[ROCMeter(),], use_checkpoint='latest', eval_interval=1)

    trainer.train(train_loader, valid_loader, 5)
