# Define Datasets

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from glob import glob
import scipy.io
import numpy as np
import json
import re

from config import AN_PATH, TRAIN_PATH, OUT_PATH, VAL_PATH, TEST_PATH, CPU_COUNT
from transformations import train_transforms, val_transforms, norm_transforms
from utils import calc_norm_parameters

class pgc_dataset(Dataset):
    """
    
    """

    def __init__(self,
                 img_paths,
                 mask_paths,
                 transforms,
                 scale_params,
                 val_set=True
                 ):
        self.scale_params = scale_params # 
        self.transforms = transforms(means=list(self.scale_params['means']), std=list(self.scale_params['std'])) # Initialize the main dataset augmentations function

        self.id_set = sorted(list({path.split('/')[-1][:-9] for path in mask_paths}.intersection({path.split('/')[-1][:-4] for path in img_paths}))) # Get set of common ids

        if val_set == True:
            self.set_aside = self.id_set[-6:]
            self.id_set = self.id_set[:-6]
            self.set_aside_paths = sorted([path for path in img_paths if path.split('/')[-1][:-4] in self.id_set])
            # self.mask_paths = sorted([path for path in mask_paths if path.split('/')[-1][:-9] in self.id_set])

        self.img_paths = sorted([path for path in img_paths if path.split('/')[-1][:-4] in self.id_set])
        self.mask_paths = sorted([path for path in mask_paths if path.split('/')[-1][:-9] in self.id_set])
        
    def __len__(self):
        return len(self.id_set)
    
    def __getitem__(self, index) -> any:
        img = cv2.imread(self.img_paths[index], cv2.COLOR_BGR2RGB)
        mask = scipy.io.loadmat(self.mask_paths[index])['data']
        # n_img = self.norm_transforms(image=img) # Normalize the image
        # n_img = n_img['image']
        transformed = self.transforms(image=img, mask=mask)
        t_img = transformed['image']
        t_mask = transformed['mask']

        return t_img, t_mask


if __name__ == '__main__':
    if not os.path.exists(os.path.join(OUT_PATH, 'scale_params/scale_params.json')):
        print(f"Calculating imageset scale parameters with {CPU_COUNT} concurrent processes")
        mean_dict, mean_array = calc_norm_parameters(directory='./data/images/', processes=CPU_COUNT)
        print("Finished!")
    else:
        with open(os.path.join(OUT_PATH, 'scale_params/scale_params.json'), 'r') as f:
            mean_dict = json.load(f)
            print("Loaded image scale parameters")

    img_paths = [i for i in glob(TRAIN_PATH+'/*') if i.endswith('jpg')]
    mask_paths = [i for i in glob(AN_PATH+'/*') if i.endswith('.mat')]

    train_dataset = pgc_dataset(img_paths=img_paths, 
                                mask_paths=mask_paths, 
                                transforms=train_transforms, 
                                scale_params=mean_dict)
    
    
    img, mask = train_dataset[13]
    print(train_dataset.id_set[13])
    print(img.shape, mask.shape)
    print(type(img), type(mask))
    print(img, mask)


