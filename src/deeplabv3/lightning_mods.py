# Pytorch Lightning Code

from typing import Any, Optional, Union
import os
import json
from glob import glob
import torch
import lightning
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from torch.optim.optimizer import Optimizer
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from model import create_deeplab_model
from config import LR, NUM_CLASSES, OUT_PATH, TRAIN_PATH, AN_PATH
from utils import collate_fn
import torch.nn as nn
from datasets import pgc_dataset
from transformations import train_transforms

torch.set_float32_matmul_precision('high')


class LitDeepLab(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.deeplab = model

    def forward(self, x) -> Any:
        return self.deeplab(x)
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.deeplab.parameters(), lr=LR)
        # scheduler = torch
        return optimizer
    
    # def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
    #     return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = nn.CrossEntropyLoss(out, mask)
        self.log("train_loss", loss)

        return loss

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

deeplab_model = create_deeplab_model("mobilenet_v3", num_classes=NUM_CLASSES, pretrained=True)

train_dataset = pgc_dataset(img_paths=img_paths, mask_paths=mask_paths, transforms=train_transforms, scale_params=mean_dict)

train_loader = DataLoader(train_dataset, batch_size=4, num_workers=8, collate_fn=collate_fn)
for v in train_loader:
    print(len(v))
    print(type(v))

# DeepLab = LitDeepLab(model=deeplab_model)

trainer = lightning.Trainer(limit_train_batches=100, max_epochs=5)
trainer.fit(model=DeepLab, train_dataloaders=train_loader)

# try:
#     next(iter(train_loader))
# except StopIteration:
#     print('finished!')

