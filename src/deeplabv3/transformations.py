# Image transformations
import albumentations as A
import numpy as np
import torch
import cv2
from albumentations.pytorch import ToTensorV2
import scipy

def train_transforms():
    """
    Returns a training dataset transformation Albumentations function
    Which performs random flips, color jitter, blurring, rotation, transposition, and distortion
    """
    transforms = A.Compose([
    A.Flip(),
    A.ColorJitter(),
    A.GaussianBlur(),
    A.SafeRotate(),
    A.Transpose(),
    A.GridDistortion(),
    ToTensorV2(p=1.0)
    ])

    return transforms

def val_transforms():
    """
    Returns a validation dataset transformation Albumentations function
    Which performs random flips, color jitter, blurring, rotation, transposition, and distortion
    """
    transforms = A.Compose([
    ToTensorV2(p=1.0)
    ])

    return transforms

def norm_transforms(means: list, std: list):
    """
    Transform to normalize image according to imageset means and standard deviations of each channel
    """
    transforms = A.Compose([
        A.Normalize(
            mean=means,
            std=std,
            always_apply=True
        )
    ])
    return transforms

if __name__ == '__main__':


    transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ])

    img = np.random.randint(low=0, high=255, size=(1000, 1000, 3)).astype(np.uint8)

    # print(img)
    # print(img.shape)

    transform(image=img)
    training = train_transforms()
    training(image=img)

    normalize = norm_transforms(means=[.10, .3, .7], std=[.23, .45, .21])
    x = transform(image=img)
    # print(x)
    # print(np.mean(x['image'], (0,1)))

    dmat = scipy.io.loadmat('./data/annotations/0afc4be3-2ebb-4c37-9560-bc77d502100f_mask.mat')
    raw_mask = dmat['data']
    print(np.unique(raw_mask))
    x = training(image = raw_mask, mask = raw_mask)
    t_mask = x['mask'].numpy()
    print(np.unique(t_mask))
