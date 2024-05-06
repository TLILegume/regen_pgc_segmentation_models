# Main model training

from utils import calc_norm_parameters
from glob import glob
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import SCALE_PARAM_PATH, TRAIN_PATH, VAL_PATH, AN_PATH, OUT_PATH, N_EPOCHS, NUM_CLASSES, BATCH_SIZE, LR, CLASS_MAPPING, GAMMA
from datasets import pgc_dataset
from transformations import train_transforms, val_transforms
from model import create_deeplab_model
import lightning.pytorch as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# check for precalculated normalization parameters

if os.path.exists(SCALE_PARAM_PATH):
    try:
        with open(SCALE_PARAM_PATH, 'r') as f:
            mean_dict = json.load(f)
        for key in ['means', 'std']:
            assert(key in mean_dict.keys())
    except AssertionError:
        print('Scale params file missing key elements. Recalculating imageset scale parameters - this may take a few minutes.')    
        mean_dict, mean_array = calc_norm_parameters('./data/images')

        with open(SCALE_PARAM_PATH, 'w') as f:
            json.dump(scale_params, f)
elif not os.path.exists(SCALE_PARAM_PATH):
    print('Scale params file missing. Calculating imageset scale parameters - this may take a few minutes.')    
    mean_dict, mean_array = calc_norm_parameters('./data/images')

    with open(SCALE_PARAM_PATH, 'w') as f:
        json.dump(mean_dict, f)
print("Imageset scale parameters loaded.")

# Grab paths
img_paths = [i for i in glob(TRAIN_PATH+'/*') if i.endswith('jpg')]
mask_paths = [i for i in glob(AN_PATH+'/*') if i.endswith('.mat')]

train_dataset = pgc_dataset(
    img_paths=img_paths, 
    mask_paths=mask_paths, 
    transforms=train_transforms, 
    scale_params=mean_dict,
    val_set=True
)

val_paths = [os.path.join('./data/images/train', i+'.jpg') for i in train_dataset.set_aside]
print(len(val_paths))

    
val_dataset = pgc_dataset(
    img_paths=val_paths, 
    mask_paths=mask_paths,
    transforms=val_transforms,
    scale_params=mean_dict,
    val_set=False
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    drop_last=True
)

model = create_deeplab_model(
    backbone='mobilenet_v3', 
    num_classes=NUM_CLASSES, 
    pretrained=True
)

model.to('cuda')

optimizer = Adam(
    params=model.parameters(),
    lr=LR
)

scheduler = ExponentialLR(
    optimizer=optimizer, 
    gamma=GAMMA
)
train_loss = []
val_loss = []

for i in range(N_EPOCHS):
    prog_bar = tqdm(train_loader, total=len(train_loader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    train_running_loss = 0
    print(scheduler.get_last_lr())
    for j, data in enumerate(prog_bar):
        optimizer.zero_grad()
        model.train()
        img, mask = data
        img = img.to('cuda')
        mask = mask.long().to('cuda')
        outputs = model(img)['out']

        # print(outputs.shape)

        # outputs = outputs.argmax(1) # get the highest unnormalized class probabilities
        # for i in CLASS_MAPPING.keys():
        #     outputs[torch.where(outputs==int(i))] = CLASS_MAPPING.get(i)
        # F.cross_entropy(input, target)
        # F.cross_entropy(img, target)
        # print(outputs.shape, mask.shape)
        criterion = CrossEntropyLoss()
        loss_value = criterion(input=outputs, target=mask)
        train_running_loss += loss_value.item()
        # print(loss_value)
        prog_bar.set_description(desc=f"Train Loss: {loss_value: .4f}")
        loss_value.backward()
        optimizer.step()

    # scheduler.step()
    print(f"Epoch {i} avg_train_loss: {train_running_loss/len(train_loader)}")
    

    prog_bar = tqdm(val_loader, total=len(val_loader), bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    val_running_loss = 0
    with torch.no_grad():
        model.eval()
        for j, data in enumerate(prog_bar):
            img, mask = data
            img = img.to('cuda')
            mask = mask.long().to('cuda')
            outputs = model(img)['out']
            criterion = CrossEntropyLoss()
            loss_value = criterion(input=outputs, target=mask)
            val_running_loss += loss_value.item()
            prog_bar.set_description(desc=f"Train Loss: {loss_value: .4f}")
    
    print(f"Epoch {i} avg_val_loss: {val_running_loss/len(val_loader)}")
    
    train_loss.append(train_running_loss/len(train_loader))
    val_loss.append(val_running_loss/len(val_loader))
    fig = plt.figure()
    plt.plot(train_loss, color='red')
    plt.plot(val_loss, color='blue')
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig(f'loss_plot_epoch_{i}.jpg')
    plt.close('all')

torch.save(model.state_dict(), './first_run.pt')

