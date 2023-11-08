# Model tests

import unittest
import string
import sys
import random
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from deeplabv3.model import create_deeplab_model
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

class TestCreateModel(unittest.TestCase):
    def test_backbone_random_string(self):
        """
        Tests if random strings raise ValueErrors when initializing model
        """
        random_strings = [random.choices(string.ascii_letters, k=8) for i in range(30)]

        for i in random_strings:
            with self.assertRaises(ValueError):
                model = create_deeplab_model(backbone=i,
                                             num_classes=4,
                                             pretrained=True
                                             )
                
    def test_backbone_strings(self):
        """
        Tests backbone values for model initialzations
        """
        backbones = ['mobilenet_v3', 'resnet50', 'resnet101']
                
        for i in backbones:
            model = create_deeplab_model(backbone=i,
                                         num_classes=4,
                                         pretrained=True)
            self.assertTrue(type(model)==DeepLabV3)


if __name__ == '__main__':
    unittest.main()
