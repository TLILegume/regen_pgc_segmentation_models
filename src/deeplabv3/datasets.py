# Define Datasets

import torch
from torch.utils.data import Dataset
import cv2
import os
from glob import glob
import scipy.io
import numpy as np

from config import AN_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, CPU_COUNT
from transformations import train_transforms, val_transforms, norm_transforms
from utils import calc_norm_parameters

class pgc_dataset(Dataset):
    """
    
    """

    def __init__(self,
                 img_paths,
                 mask_paths,
                 transforms,
                 norm_transforms,
                 dataset_norm
                 ):
        self.img_paths = img_paths
        self.img_ids = None
        self.mask_paths = mask_paths
        self.mask_ids = None
        self.dataset_norm = dataset_norm
        self.transforms = transforms()
        self.norm_transforms = norm_transforms(means=list(self.dataset_norm['means']), std=list(self.dataset_norm['std']))

        print(self.img_paths)
        print(self.mask_paths)
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index) -> any:
        img = cv2.imread(self.img_paths[index], cv2.COLOR_BGR2RGB)
        mask = scipy.io.loadmat(self.mask_paths[index])
        mask = mask['data']

        n_img = self.norm_transforms(image=img)
        print(type(n_img))
        n_img = n_img['image']
        print(type(n_img))

        print(mask.shape)
        print(n_img.shape)
        transformed = self.transforms(image=n_img, mask=mask)
        t_img = torch.tensor(np.transpose(transformed['image'].numpy(), (2, 0, 1)), dtype=torch.float)
        t_mask = torch.squeeze(transformed['mask'])

        return t_img, t_mask


        pass


if __name__ == '__main__':
    mean_dict = calc_norm_parameters(directory='./data/images/', processes=CPU_COUNT)
    print(mean_dict)

    img_paths = [i for i in glob(TRAIN_PATH+'/*') if i.endswith('jpg')]
    mask_paths = [i for i in glob(AN_PATH+'/*') if i.endswith('.mat')]
    print(mask_paths)

    train_dataset = pgc_dataset(img_paths=img_paths, 
                                mask_paths=mask_paths, 
                                transforms=train_transforms, 
                                norm_transforms=norm_transforms,
                                dataset_norm=mean_dict)

    print(train_dataset[0])
    print(train_dataset.img_paths)
    print(train_dataset.mask_paths)

