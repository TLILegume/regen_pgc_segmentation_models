# Model creation

import torch
import cv2
import torchvision
import torch.nn as nn
import numpy as np
from torchvision.io import read_image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# from config import NUM_CLASSES

def create_deeplab_model(backbone: str, num_classes: int, pretrained: bool=True) -> torchvision.models.segmentation.deeplabv3:
    """
    
    """
    model_backbones = {
        'mobilenet_v3': deeplabv3_mobilenet_v3_large,
        'resnet50': deeplabv3_resnet50,
        'resnet101': deeplabv3_resnet101
    }
    if backbone not in list(model_backbones.keys()):
        raise ValueError(f"Model backbone should be string with a value in {list(model_backbones.keys())}")
    
    model = model_backbones[backbone](weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    
    return model


if __name__ == '__main__':

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    transforms = weights.transforms(resize_size=None)
    
    img = read_image('./data/images/puppy.jpg')
    batch = torch.stack([transforms(img)])
    print(type(img))
    print(img.shape)
    model.eval()

    out = model(batch)['out']

    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

    normalized_masks = torch.nn.functional.softmax(out, dim=1)

    dog_and_boat_masks = [
        normalized_masks[img_idx, sem_class_to_idx[cls]]
        for img_idx in range(len([img]))
        for cls in ('dog', 'boat')
    ]

    show(dog_and_boat_masks)