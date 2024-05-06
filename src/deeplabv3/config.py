# Pytorch configuration settings
from multiprocessing import cpu_count
from glob import glob

# Directories
TRAIN_PATH = './data/images/train'        # Path to training images directory
VAL_PATH = './data/images/val'            # Validation image directory
TEST_PATH = './data/images/testing'       # Testing image directory
AN_PATH = './data/annotations'            # Annotation files directory
OUT_PATH = './outputs'                    # Model output root directory
LOG_PATH = './logs'                       # Log files directory

img_paths = [i for i in glob(TRAIN_PATH+'/*') if i.endswith('jpg')]
mask_paths = [i for i in glob(AN_PATH+'/*') if i.endswith('.mat')]



SCALE_PARAM_PATH = './outputs/scale_params/scale_params.json'

# CPU parameters
CPU_COUNT = cpu_count()

# Model parameters
N_EPOCHS = 30
LR = 0.001
MOM = 0.9
BETA = 0.05
GAMMA = 0.98
BATCH_SIZE = 2

# Data parameters
NUM_CLASSES = 8
CLASSES = {
    'background': 0,
    'pgc_grass_mask': 1,
    'pgc_clover_mask': 2,
    'soil_mask': 3,
    'weed_mask': 4,
    'residue_mask': 5,
    'corn_mask': 6,
    'soybean_mask': 7
}

MASK_CLASSES = {
    'background': 0,
	'pgc_grass_mask': 1,
	'pgc_clover_mask': 2,
	'soil_mask': 3,
	'weed_mask': 4,
	'residue_mask': 5,
	'corn_mask': 6,
	'soybean_mask': 7
}

CLASS_MAPPING = {
    '0': 0,
    '1': 50,
    '2': 75,
    '3': 100,
    '4': 150,
    '5': 175,
    '6': 200
}

BBOX_CLASSES = {
	'0': 'background',
	'1': 'blue_marker',
	'2': 'pink_marker',
	'3': 'orange_marker',
	'4': 'yellow_marker'
}