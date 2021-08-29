import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
import cv2

#!pip install nnAudio
from nnAudio.Spectrogram import CQT1992v2

from utils import convert_image_id_2_path

class DataRetriever(torch.utils.data.Dataset):
    def __init__(self, paths, targets):
        self.paths = paths
        self.targets = targets
        
        self.q_transform = CQT1992v2(
            sr=2048, fmin=20, fmax=1024, hop_length=16, bins_per_octave=16,
        )
          
    def __len__(self):
        return len(self.paths)
    
    def __get_qtransform(self, x):
        image = []
        for i in range(3):
            waves = x[i] / np.max(x[i])
            waves = torch.from_numpy(waves).float()
            channel = self.q_transform(waves).squeeze().numpy()
            image.append(channel)
            
        return np.stack(image, axis=-1) # [H, W, 3]
    
    def __getitem__(self, index):
        file_path = convert_image_id_2_path(self.paths[index])
        x = np.load(file_path)
        image = self.__get_qtransform(x)

        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)

        # aug
        if np.random.uniform() > 0.5:
            image = np.flip(image, 0)
        if np.random.uniform() > 0.5:
            image = np.flip(image, 1)

        image = np.ascontiguousarray(image)
        
        image = torch.from_numpy(image).permute(2,0,1).float()
        y = torch.tensor(self.targets[index], dtype=torch.float)
            
        return {"X": image, "y": y}

class TestDataRetriever(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

        self.q_transform = CQT1992v2(
            sr=2048, fmin=20, fmax=1024, hop_length=32
        )
          
    def __len__(self):
        return len(self.paths)
    
    def __get_qtransform(self, x):
        image = []
        for i in range(3):
            waves = x[i] / np.max(x[i])
            waves = torch.from_numpy(waves).float()
            channel = self.q_transform(waves).squeeze().numpy()
            image.append(channel)
            
        return np.stack(image, axis=-1) # [H, W, 3]
    
    def __getitem__(self, index):
        file_path = convert_image_id_2_path(self.paths[index], is_train=False)
        x = np.load(file_path)
        image = self.__get_qtransform(x)
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).permute(2,0,1).float()
            
        return {"X": image, "id": self.paths[index]}

