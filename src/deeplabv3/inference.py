# Inference

import torch
import json
import os
from glob import glob
import cv2
import albumentations as A
from model import create_deeplab_model
from config import NUM_CLASSES, TRAIN_PATH, SCALE_PARAM_PATH, AN_PATH, TRAIN_PATH, BATCH_SIZE
from datasets import pgc_dataset
from albumentations.pytorch import ToTensorV2
import numpy as np
from transformations import val_transforms
from torch.utils.data import DataLoader
from numpy.random import randint

# Model State Path
chkpt_path = './first_run.pt'

# Image path
img_path = './data/images/train/0de13bf8-9122-47e2-9d96-4065c49a63d2.jpg'

# Read in scale parameters
with open(SCALE_PARAM_PATH, 'r') as f:
    mean_dict = json.load(f)

# Grab paths
img_paths = [i for i in glob(TRAIN_PATH+'/*') if i.endswith('jpg')]
mask_paths = [i for i in glob(AN_PATH+'/*') if i.endswith('.mat')]
    
val_dataset = pgc_dataset(
    img_paths=img_paths, 
    mask_paths=mask_paths,
    transforms=val_transforms,
    scale_params=mean_dict
)


val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=16
)

model = create_deeplab_model(
    backbone='mobilenet_v3', 
    num_classes=NUM_CLASSES, 
    pretrained=True
)

model.load_state_dict(torch.load(chkpt_path))
model.eval()
for j in range(len(val_dataset)):
    img, mask = val_dataset[j]
    img_id = val_dataset.id_set[j]
    print(img_id)
    img = torch.unsqueeze(img, 0)
    output = model(img)['out']

    classed = torch.argmax(output.squeeze(0), 0)

    mask_array = classed.numpy()

    raw_img = cv2.imread(os.path.join(TRAIN_PATH, img_id+'.jpg'))
    for i in [1, 3, 4]:
        class_mask = np.expand_dims(np.where(mask_array==i, 1, 0), 2)
        class_mask = np.repeat(class_mask, 3, 2)
        masked = np.ma.MaskedArray(raw_img, mask=class_mask, fill_value=(255, 0, 0))
        image_overlay = masked.filled()
        
        class_mask = class_mask.astype(np.uint8)
        cv2.imwrite(f"class_{i}.jpg", class_mask)
        # print(class_mask.shape, class_mask.min(), class_mask.max())
        # print(raw_img.dtype, class_mask.dtype)
        # raw_img = cv2.addWeighted(class_mask, .7, raw_img, .3, 0)
        combined = cv2.addWeighted(raw_img, 1 - .3, image_overlay, .3, 0)
        cv2.imwrite(f'{img_id}_class_{i}.jpg', combined)
    # print(img.shape)