# Pytorch configuration settings
from multiprocessing import cpu_count

# Directories
TRAIN_PATH = './data/images/train'        # Path to training images directory
VAL_PATH = './data/images/val'            # Validation image directory
TEST_PATH = './data/images/testing'       # Testing image directory
AN_PATH = './data/annotations'            # Annotation files directory
OUT_PATH = './outputs'                    # Model output root directory
LOG_PATH = './logs'                       # Log files directory

SCALE_PARAM_PATH = './outputs/scale_params/scale_params.json'

# CPU parameters
CPU_COUNT = cpu_count()
# Model parameters

# Data parameters
NUM_CLASSES = 7
CLASSES = {
      'background': 0,
      'pgc_grass_mask': 1,
      'pgc_clover_mask': 2,
      'soil_mask': 2,
      'weed_mask': 3,
      'residue_mask': 4,
      'corn_mask': 5,
      'soybean_mask': 6
}

MASK_CLASSES = {
	'pgc_grass_mask': 50,
	'pgc_clover_mask': 75,
	'soil_mask': 100,
	'weed_mask': 125,
	'residue_mask': 150,
	'corn_mask': 175,
	'soybean_mask': 200
}

BBOX_CLASSES = {
	'0': 'background',
	'1': 'blue_marker',
	'2': 'pink_marker',
	'3': 'orange_marker',
	'4': 'yellow_marker'
}