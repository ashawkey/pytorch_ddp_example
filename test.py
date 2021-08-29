import os
import json
import random
import collections

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils import consume_prefix_in_state_dict_if_present

#!pip install nnAudio
from nnAudio.Spectrogram import CQT1992v2

import time

from torch import nn

#!pip install efficientnet_pytorch
import efficientnet_pytorch

from sklearn.model_selection import StratifiedKFold

from utils import *
from models import Model
from provider import TestDataRetriever as DataRetriever

SEED = 42

seed_everything(SEED)


submission = pd.read_csv("../input/g2net-gravitational-wave-detection/sample_submission.csv")

test_data_retriever = DataRetriever(
    submission["id"].values, 
)

test_loader = torch.utils.data.DataLoader(
    test_data_retriever,
    batch_size=128,
    shuffle=False,
    pin_memory=True,
    num_workers=8,
)

checkpoints = [
    'workspace/checkpoints/fold0.pth.tar',
    'workspace/checkpoints/fold1.pth.tar',
    'workspace/checkpoints/fold2.pth.tar',
    'workspace/checkpoints/fold3.pth.tar',
    'workspace/checkpoints/fold4.pth.tar',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.to(device).eval()

preds = []
ids = []

for fold, checkpoint in enumerate(checkpoints):
    model.load_state_dict(consume_prefix_in_state_dict_if_present(torch.load(checkpoint)['model']))
    pred = []
    for e, batch in enumerate(test_loader):
        print(f"[eval] model {fold} | {e+1}/{len(test_loader)}", end="\r")
        with torch.no_grad():
            y = model(batch["X"].to(device)) # [B]
            pred.append(y)
            if fold == 0:
                ids.extend(batch["id"])

    pred = torch.cat(pred, dim=0)
    submission = pd.DataFrame({"id": ids, "target": torch.sigmoid(pred).cpu().numpy()})
    submission.to_csv(f"workspace/submission_fold{fold}.csv", index=False)

    preds.append(pred)

# mean then sigmoid...
preds = torch.mean(torch.stack(preds, axis=0), axis=0) # [folds, N] -> [N]
preds = torch.sigmoid(preds)
preds = preds.cpu().numpy()

submission = pd.DataFrame({"id": ids, "target": preds})
submission.to_csv("workspace/submission.csv", index=False)
