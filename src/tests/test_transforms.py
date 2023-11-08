# Transformation tests

import unittest
import sys
from pathlib import Path
import numpy as np
from numpy.random import randint
import torch

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from deeplabv3.transformations import train_transforms, val_transforms

class TestTransforms(unittest.TestCase):
    def test_train_transforms(self):
        for i in range(100):
            img = np.random.randint(low=0, high=255, size=(1000, 1000, 3)).astype(np.uint8)
            mask = np.zeros((1000, 1000, 4))
            for i in range(4):
                mask[:200,:200,i] = randint(low=1, high=255)
                np.random.shuffle(mask[:, :, i])

            transformed = train_transforms(image=img, mask=mask)
            
            self.assertIsInstance(transformed['image'], torch.Tensor)
            self.assertIsInstance(transformed['mask'], torch.Tensor)
            self.assertLessEqual(torch.max(transformed['image']), 255)
            
    
    def test_val_transforms(self):
        for i in range(100):
            img = np.random.randint(low=0, high=255, size=(1000, 1000, 3)).astype(np.uint8)
            mask = np.zeros((1000, 1000, 4))
            for i in range(4):
                mask[:200,:200,i] = randint(low=1, high=255)
                np.random.shuffle(mask[:, :, i])
            print(mask)
            transformed = train_transforms(image=img, mask=mask)
            
            self.assertIsInstance(transformed['image'], torch.Tensor)
            self.assertIsInstance(transformed['mask'], torch.Tensor)
            self.assertLessEqual(torch.max(transformed['image']), 255)
    
if __name__ == '__main__':
    unittest.main()